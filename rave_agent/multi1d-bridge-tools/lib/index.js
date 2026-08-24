// multi1d-bridge-tools — DSH plugin: bridge Multi1D++ ASCII output into
// RAVE-SIM grid inputs.
//
// This is a static bootstrap plugin (same pattern as rave-sim-tools). It
// registers four model-facing tools directly via `tools.register`:
//
//   multi1d_load        — load a Multi1D++ ASCII output, return a summary
//   multi1d_build_grids — turn one timestep's 1D profile into RAVE-SIM
//                         PlasmaSample / precise_Sample grids (saved as .npy)
//   multi1d_gen_config  — additionally write a RAVE-SIM config.yaml
//   multi1d_grid_plot   — render a grid (ne/te/zstar or material/density) to a
//                         PNG under output/_agent_runs/plots + markdown URL
//
// All business logic lives in the Python bridge modules; this file only
// spawns the interpreter and returns the JSON. Kept thin and restart-stable.
//
// NOTE: parameters must be full JSON Schema ({type:'object', ...}) — the model
// catalog stores them raw (no spec compilation), same as rave-sim-tools.

const PYTHON = '/home/taylor/anaconda3/envs/rave-sim/bin/python'
const BRIDGE_SCRIPT = '/mnt/d/rave-sim-main/rave-sim-main/rave_agent/multi1d_bridge.py'
const GRID_PLOT_SCRIPT = '/mnt/d/rave-sim-main/rave-sim-main/rave_agent/multi1d_grid_plot.py'

export default {
  name: 'multi1d-bridge-tools',
  inject: ['fs', 'tools', 'subprocess'],
  apply(ctx) {
    const tools = ctx.tools
    const fsService = ctx.fs

    function renderText(_args, value) {
      return [{ type: 'text', text: JSON.stringify(value, null, 2) }]
    }

    function requireArgs(args, keys) {
      for (const k of keys) {
        if (args[k] === undefined || args[k] === null || String(args[k]) === '') {
          return k + ' is required'
        }
      }
      return null
    }

    async function runScript(argv, maxBytes) {
      const sub = ctx.get('subprocess')
      if (sub === undefined) return { error: 'subprocess service unavailable' }
      let h
      try {
        h = sub.spawn({
          argv, cwd: '/', graceMs: 5000,
          stdio: {
            stdin: 'ignore',
            stdout: { maxBytes: maxBytes || 65536, spill: { maxBytes: 1 << 20 } },
            stderr: { maxBytes: maxBytes || 65536, spill: { maxBytes: 1 << 20 } },
          },
        })
      } catch (e) {
        return { error: 'spawn failed: ' + String(e && e.message ? e.message : e) }
      }
      let outcome
      try { outcome = await h.done } catch (e) { outcome = { exitCode: -1, signal: null } }
      const out = h.collected.stdout ? h.collected.stdout.readFrom(0).text : ''
      const err = h.collected.stderr ? h.collected.stderr.readFrom(0).text : ''
      return { outcome, out, err }
    }

    async function runBridge(cmd, payload) {
      const r = await runScript([PYTHON, BRIDGE_SCRIPT, cmd, JSON.stringify(payload)], 65536)
      const text = r.out.trim()
      if (text) {
        try { return JSON.parse(text) } catch (e) { /* fall through */ }
      }
      return { error: 'bridge exited ' + r.outcome.exitCode + ': ' + ((r.err || text).slice(0, 500) || '(no output)') }
    }

    async function runGridPlot(payload) {
      const argv = [PYTHON, GRID_PLOT_SCRIPT, '--grid_dir', payload.grid_dir]
      if (payload.grid) argv.push('--grid', payload.grid)
      if (payload.out) argv.push('--out', payload.out)
      if (payload.fields) argv.push('--fields', payload.fields)
      if (payload.pixel_size_x_um != null) argv.push('--pixel_size_x_um', String(payload.pixel_size_x_um))
      if (payload.pixel_size_z_um != null) argv.push('--pixel_size_z_um', String(payload.pixel_size_z_um))
      const r = await runScript(argv, 65536)
      const text = r.out.trim()
      if (text) {
        try { return JSON.parse(text) } catch (e) { /* fall through */ }
      }
      return { error: 'grid plot exited ' + r.outcome.exitCode + ': ' + ((r.err || text).slice(0, 500) || '(no output)') }
    }

    // ── multi1d_load ──────────────────────────────────────────────────────
    tools.register({
      name: 'multi1d_load',
      description: 'Load a Multi1D++ radiation-hydrodynamics ASCII output (.scalars.dat / .center.dat) and return a summary: number of timesteps, cells, groups, time/XC/rho/Te ranges, material count, and available timesteps. Use this first to inspect a case before building grids.',
      parameters: {
        type: 'object',
        properties: {
          case_path: { type: 'string', description: 'Multi1D++ case file prefix or output directory, e.g. /path/to/251222 or /path/to/251222test' },
          parse_materials: { type: 'boolean', description: 'Parse matter++/material.base into a material map (default true)' },
        },
        required: ['case_path'],
      },
      output: {
        schema: { type: 'object', properties: { ok: { type: 'boolean' }, nt: { type: 'number' }, ncell: { type: 'number' }, ngroups: { type: 'number' }, n_materials: { type: 'number' } }, additionalProperties: true },
        render: renderText,
      },
      async execute(args) {
        const missing = requireArgs(args, ['case_path'])
        if (missing) return { error: missing }
        return runBridge('load', { case_path: args.case_path, parse_materials: args.parse_materials !== false })
      },
    })

    // ── multi1d_build_grids ───────────────────────────────────────────────
    tools.register({
      name: 'multi1d_build_grids',
      description: 'Convert one timestep of a Multi1D++ 1D profile into RAVE-SIM grids and save them as .npy: PlasmaSample four-grid (ne/ni/te/zstar_grid.npy) for side-on/face-on, plus a precise_Sample material_grid/density_grid for the cold region when face-on. Returns per-grid shape and ne/te/zstar ranges. This is the core bridge output consumed by RAVE-SIM.',
      parameters: {
        type: 'object',
        properties: {
          case_path: { type: 'string', description: 'Multi1D++ case prefix or output directory' },
          timestep: { type: 'number', description: '0-based timestep index (default: last timestep)' },
          output_dir: { type: 'string', description: 'Directory to write the grid .npy files into (created if missing)' },
          geometry: { type: 'string', enum: ['side-on', 'face-on'], description: 'side-on: laser ⊥ X-ray (profile → x axis, 1 z layer); face-on: laser ∥ X-ray (profile → z axis)' },
          nx: { type: 'number', description: 'Transverse pixels (default: 256)' },
          transverse_size_um: { type: 'number', description: 'Transverse physical extent in µm (default: 200)' },
          los_thickness_um: { type: 'number', description: 'Line-of-sight thickness in µm for side-on (default: 100)' },
          pixel_size_z_um: { type: 'number', description: 'Z pixel in µm for face-on (default: 1)' },
          max_nz: { type: 'number', description: 'Max z-layers for face-on (default: 500)' },
        },
        required: ['case_path', 'output_dir'],
      },
      output: {
        schema: { type: 'object', properties: { ok: { type: 'boolean' }, output_dir: { type: 'string' }, paths: { type: 'object' }, grids: { type: 'object' } }, additionalProperties: true },
        render: renderText,
      },
      async execute(args) {
        const missing = requireArgs(args, ['case_path', 'output_dir'])
        if (missing) return { error: missing }
        return runBridge('build_grids', {
          case_path: args.case_path,
          timestep: args.timestep,
          output_dir: args.output_dir,
          geometry: args.geometry,
          nx: args.nx,
          transverse_size_um: args.transverse_size_um,
          los_thickness_um: args.los_thickness_um,
          pixel_size_z_um: args.pixel_size_z_um,
          max_nz: args.max_nz,
        })
      },
    })

    // ── multi1d_gen_config ────────────────────────────────────────────────
    tools.register({
      name: 'multi1d_gen_config',
      description: 'From a Multi1D++ case, build the grids for a timestep AND write a RAVE-SIM config.yaml (plus the .npy files) into output_dir. Supports 1D/2D, 8–10 keV default source, and hybrid plasma_sample + precise_sample elements. Verifies the config reloads through RAVE-SIM own config loader.',
      parameters: {
        type: 'object',
        properties: {
          case_path: { type: 'string', description: 'Multi1D++ case prefix or output directory' },
          timestep: { type: 'number', description: '0-based timestep index (default: last)' },
          output_dir: { type: 'string', description: 'Directory for the .npy grids + config.yaml' },
          geometry: { type: 'string', enum: ['side-on', 'face-on'], description: 'Grid geometry (default: side-on)' },
          dimension: { type: 'string', enum: ['1d', '2d'], description: 'RAVE-SIM simulation dimension (default: 1d)' },
          nx: { type: 'number', description: 'Transverse pixels (default: 256)' },
          transverse_size_um: { type: 'number', description: 'Transverse extent µm (default: 200)' },
          los_thickness_um: { type: 'number', description: 'LOS thickness µm for side-on (default: 100)' },
          pixel_size_z_um: { type: 'number', description: 'Z pixel µm for face-on (default: 1)' },
          energy_min_eV: { type: 'number', description: 'X-ray energy lower bound eV (default: 8000)' },
          energy_max_eV: { type: 'number', description: 'X-ray energy upper bound eV (default: 10000)' },
          z_target_m: { type: 'number', description: 'Source-to-target distance m (default: 0.5)' },
          z_detector_m: { type: 'number', description: 'Source-to-detector distance m (default: 4.8)' },
          N: { type: 'number', description: 'Wavefront sample points (default: 33554432; small e.g. 8192 for a fast probe)' },
          dx_m: { type: 'number', description: 'Transverse pixel m (default: 3e-10)' },
          chunk_size: { type: 'number', description: 'Chunk size (default: 16777216)' },
          nr_source_points: { type: 'number', description: 'Monte Carlo source points (default: 100)' },
        },
        required: ['case_path', 'output_dir'],
      },
      output: {
        schema: { type: 'object', properties: { ok: { type: 'boolean' }, config_path: { type: 'string' }, config_reload_ok: { type: 'boolean' }, n_elements: { type: 'number' }, element_types: { type: 'array', items: { type: 'string' } } }, additionalProperties: true },
        render: renderText,
      },
      async execute(args) {
        const missing = requireArgs(args, ['case_path', 'output_dir'])
        if (missing) return { error: missing }
        return runBridge('gen_config', {
          case_path: args.case_path,
          timestep: args.timestep,
          output_dir: args.output_dir,
          geometry: args.geometry,
          dimension: args.dimension,
          nx: args.nx,
          transverse_size_um: args.transverse_size_um,
          los_thickness_um: args.los_thickness_um,
          pixel_size_z_um: args.pixel_size_z_um,
          energy_min_eV: args.energy_min_eV,
          energy_max_eV: args.energy_max_eV,
          z_target_m: args.z_target_m,
          z_detector_m: args.z_detector_m,
          N: args.N,
          dx_m: args.dx_m,
          chunk_size: args.chunk_size,
          nr_source_points: args.nr_source_points,
        })
      },
    })

    // ── multi1d_grid_plot ─────────────────────────────────────────────────
    tools.register({
      name: 'multi1d_grid_plot',
      description: 'Render a grid produced by multi1d_build_grids / multi1d_gen_config to PNG(s) under output/_agent_runs/plots and return a markdown-embeddable http url. Default renders the plasma fields ne,te,zstar; pass --grid for a single explicit .npy (e.g. density_grid.npy). Embed the image in your reply with ![title](url).',
      parameters: {
        type: 'object',
        properties: {
          grid_dir: { type: 'string', description: 'Directory containing the grid .npy files (output_dir from build_grids/gen_config)' },
          fields: { type: 'string', description: 'Comma-separated fields to render (default: ne,te,zstar)' },
          grid: { type: 'string', description: 'Explicit .npy path to render a single field (overrides fields)' },
          out: { type: 'string', description: 'Optional output PNG path (single grid only)' },
          pixel_size_x_um: { type: 'number', description: 'Override transverse pixel size in µm' },
          pixel_size_z_um: { type: 'number', description: 'Override Z pixel size in µm' },
        },
        required: ['grid_dir'],
      },
      output: {
        schema: { type: 'object', properties: { ok: { type: 'boolean' }, sim_dir: { type: 'string' }, grids: { type: 'array', items: { type: 'object', additionalProperties: true } } }, additionalProperties: true },
        render: renderText,
      },
      async execute(args) {
        const missing = requireArgs(args, ['grid_dir'])
        if (missing) return { error: missing }
        return runGridPlot({
          grid_dir: args.grid_dir,
          fields: args.fields,
          grid: args.grid,
          out: args.out,
          pixel_size_x_um: args.pixel_size_x_um,
          pixel_size_z_um: args.pixel_size_z_um,
        })
      },
    })

    console.log('[multi1d-bridge-tools] registered: multi1d_load, multi1d_build_grids, multi1d_gen_config, multi1d_grid_plot')
  },
}
