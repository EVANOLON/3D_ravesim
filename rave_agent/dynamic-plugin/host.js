// RAVE-SIM GPU simulation plugin v10: 7 tools + inline plot/grid viewer RPC
// (dynamic-plugin host half — kept in sync with the running rave-1/pkg-1)
const PLOT_SCRIPT = '/mnt/d/rave-sim-main/rave-sim-main/rave_agent/plot_result.py'
const GRID_PLOT_SCRIPT = '/mnt/d/rave-sim-main/rave-sim-main/rave_agent/grid_plot.py'
const PLOT_SERVER_SCRIPT = '/mnt/d/rave-sim-main/rave-sim-main/rave_agent/plot_server.py'
const OUTPUT_ROOT = '/mnt/d/rave-sim-main/rave-sim-main/output'
const PLOTS_DIR = OUTPUT_ROOT + '/_agent_runs/plots'
let plotServerBase = null

function bytesToBase64(u8) {
  let bin = ''
  const chunk = 0x8000
  for (let i = 0; i < u8.length; i += chunk) {
    bin += String.fromCharCode.apply(null, u8.subarray(i, i + chunk))
  }
  return btoa(bin)
}

return {
  inject: ['subprocess', 'fs'],
  apply(ctx) {
    const subprocess = ctx.get('subprocess')
    const fsService = ctx.get('fs')
    if (subprocess === undefined || fsService === undefined) return

    const PYTHON = '/home/taylor/anaconda3/envs/rave-sim/bin/python'
    const VALIDATE_SCRIPT = '/mnt/d/rave-sim-main/rave-sim-main/rave_agent/validate_sim.py'
    const DEFAULT_FASTWAVE = '/mnt/d/rave-sim-main/rave-sim-main/fast-wave/build-Release/fastwave'
    const NVIDIA_SMI_CANDIDATES = [
      '/usr/lib/wsl/lib/nvidia-smi',
      '/usr/bin/nvidia-smi',
      '/usr/local/bin/nvidia-smi',
    ]

    function normalizePath(p) {
      const abs = String(p).startsWith('/') ? String(p) : '/' + String(p)
      const parts = []
      for (const seg of abs.split('/')) {
        if (seg === '' || seg === '.') continue
        if (seg === '..') parts.pop()
        else parts.push(seg)
      }
      return '/' + parts.join('/')
    }

    function isUnderOutputRoot(p) {
      const t = normalizePath(p)
      const root = normalizePath(OUTPUT_ROOT)
      return t === root || t.startsWith(root + '/')
    }

    function defaultTarget(simDir) {
      const base = (String(simDir).split('/').filter(Boolean).pop() || 'sim').replace(/[^A-Za-z0-9._-]/g, '_')
      const ts = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 14)
      return OUTPUT_ROOT + '/_agent_runs/' + base + '__agentrun_' + ts
    }

    function resolveDetected(simDir) {
      return String(simDir).replace(/\/+$/, '') + '/00000000/detected.npy'
    }

    const jobs = new Map()
    let seq = 0

    function spawnSync(argv, cwd, env, maxBytes) {
      return subprocess.spawn({
        argv, cwd, env, graceMs: 5000,
        stdio: {
          stdin: 'ignore',
          stdout: { maxBytes: maxBytes || 65536, spill: { maxBytes: 1 << 20 } },
          stderr: { maxBytes: maxBytes || 65536, spill: { maxBytes: 1 << 20 } },
        },
      })
    }

    async function runAndCollect(argv, cwd, env, maxBytes) {
      const h = spawnSync(argv, cwd, env, maxBytes)
      let outcome, spawnError = null
      try { outcome = await h.done } catch (e) { outcome = { exitCode: -1, signal: null }; spawnError = String(e && e.message ? e.message : e) }
      const out = h.collected.stdout ? h.collected.stdout.readFrom(0).text : ''
      const err = h.collected.stderr ? h.collected.stderr.readFrom(0).text : ''
      return { outcome, out, err, spawnError }
    }

    async function gpuInfo() {
      for (const smi of NVIDIA_SMI_CANDIDATES) {
        try {
          const r = await runAndCollect([smi, '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], '/', undefined, 2048)
          if (r.outcome.exitCode === 0) return { ok: true, line: r.out.trim() }
          if (r.spawnError) continue
          return { ok: false, detail: ((r.err || r.out).trim() || 'nvidia-smi exited ' + r.outcome.exitCode).slice(0, 200) }
        } catch (e) { return { ok: false, detail: String(e).slice(0, 200) } }
      }
      return { ok: false, detail: 'no nvidia-smi found in: ' + NVIDIA_SMI_CANDIDATES.join(', ') }
    }

    async function copySimDir(src, targetDir) {
      if (!isUnderOutputRoot(targetDir)) {
        throw new Error('copy target rejected: must be inside ' + OUTPUT_ROOT + ' (got ' + targetDir + ')')
      }
      const mkdir = await runAndCollect(['/bin/mkdir', '-p', OUTPUT_ROOT + '/_agent_runs'], '/', undefined, 4096)
      if (mkdir.outcome.exitCode !== 0) throw new Error('mkdir failed: ' + (mkdir.err || mkdir.out))
      const rm = await runAndCollect(['/bin/rm', '-rf', targetDir], '/', undefined, 4096)
      if (rm.outcome.exitCode !== 0) throw new Error('rm failed: ' + (rm.err || rm.out))
      const cp = await runAndCollect(['/bin/cp', '-r', src, targetDir], '/', undefined, 4096)
      if (cp.outcome.exitCode !== 0) throw new Error('copy failed: ' + (cp.err || cp.out))
    }

    async function runValidator(args) {
      const r = await runAndCollect([PYTHON, VALIDATE_SCRIPT, ...args], '/', undefined, 65536)
      const text = r.out.trim()
      if (text) {
        try { return JSON.parse(text) } catch (e) { /* fall through */ }
      }
      return { error: 'validator exited ' + r.outcome.exitCode + ': ' + ((r.err || text).slice(0, 500) || '(no output)') }
    }

    async function makePlot(npyPath, simDir) {
      const base = (npyPath.split('/').filter(Boolean).pop() || 'result').replace(/\.npy$/i, '')
      const out = PLOTS_DIR + '/' + base + '_' + Date.now() + '.png'
      const argv = [PYTHON, PLOT_SCRIPT, '--path', npyPath, '--out', out]
      if (simDir) argv.push('--sim_dir', simDir)
      const r = await runAndCollect(argv, '/', undefined, 65536)
      if (r.outcome.exitCode !== 0) return { error: 'plot failed: ' + ((r.err || r.out).trim().slice(0, 300) || 'exit ' + r.outcome.exitCode) }
      try { return JSON.parse(r.out.trim()) } catch (e) { return { error: 'plot produced invalid JSON: ' + String(e) } }
    }

    async function makeGridPlot(simDir, gridPath) {
      const argv = [PYTHON, GRID_PLOT_SCRIPT, '--sim_dir', simDir]
      if (gridPath) argv.push('--grid', gridPath)
      const r = await runAndCollect(argv, '/', undefined, 65536)
      if (r.outcome.exitCode !== 0) return { error: 'grid plot failed: ' + ((r.err || r.out).trim().slice(0, 300) || 'exit ' + r.outcome.exitCode) }
      try {
        const j = JSON.parse(r.out.trim())
        if (j.ok && j.grids && j.grids.length && !j.grids[0].error) {
          const g = j.grids[0]
          return { png_path: g.png_path, url: g.url, shape: g.shape, dtype: g.dtype, kind: g.kind, grids: j.grids }
        }
        return j
      } catch (e) { return { error: 'grid plot produced invalid JSON: ' + String(e) } }
    }

    // Start (or reuse) the inline-plot HTTP server; returns the base URL or ''.
    async function ensurePlotServer() {
      if (plotServerBase !== null) return plotServerBase
      const r = await runAndCollect([PYTHON, PLOT_SERVER_SCRIPT, 'start'], '/', undefined, 8192)
      try {
        const j = JSON.parse(r.out.trim())
        if (j && j.ok && j.url) { plotServerBase = j.url; return plotServerBase }
      } catch (e) { /* ignore */ }
      plotServerBase = ''
      return plotServerBase
    }

    // Attach the markdown-embeddable http URL to a plot result (png_path present).
    async function attachPlotUrl(res) {
      if (!res || res.error || !res.png_path) return res
      const base = await ensurePlotServer()
      if (base) res.url = base + '/' + (res.png_path.split('/').filter(Boolean).pop() || '')
      return res
    }

    async function npySummary(path) {
      const target = await fsService.resolve(path)
      const info = await fsService.stat(target)
      if (!info) throw new Error('file not found: ' + path)
      const u8 = await fsService.readBytes(target, undefined, 32 * 1024 * 1024)
      if (u8.length < 10) throw new Error('file too small to be npy')
      const view = new DataView(u8.buffer, u8.byteOffset, u8.byteLength)
      const magic = String.fromCharCode(u8[0], u8[1], u8[2], u8[3], u8[4], u8[5])
      if (magic !== '\x93NUMPY') throw new Error('not an npy file: ' + path)
      const major = u8[6]
      const headerLen = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true)
      const headerStart = major === 1 ? 10 : 12
      const header = new TextDecoder('utf-8').decode(u8.subarray(headerStart, headerStart + headerLen))
      const shapeMatch = header.match(/'shape':\s*\(([^)]*)\)/)
      const descrMatch = header.match(/'descr':\s*'([^']*)'/)
      const shape = shapeMatch ? shapeMatch[1].split(',').map(function (s) { return s.trim() }).filter(function (s) { return s !== '' }).map(Number) : []
      const descr = descrMatch ? descrMatch[1] : '?'
      const total = shape.reduce(function (a, b) { return a * b }, 1)
      const dataStart = headerStart + headerLen
      const isF4 = descr === '<f4' || descr === '|f4' || descr === '=f4'
      const isF8 = descr === '<f8' || descr === '|f8' || descr === '=f8'
      let min = Infinity, max = -Infinity, sum = 0, n = 0
      const head = []
      if (isF4 || isF8) {
        const size = isF4 ? 4 : 8
        const dv = new DataView(u8.buffer, u8.byteOffset + dataStart, total * size)
        for (let i = 0; i < total; i++) {
          const v = isF4 ? dv.getFloat32(i * size, true) : dv.getFloat64(i * size, true)
          if (v < min) min = v
          if (v > max) max = v
          sum += v; n++
          if (i < 8) head.push(Number(v.toPrecision(6)))
        }
      }
      return { path, shape, dtype: descr, total, min: n ? Number(min.toPrecision(6)) : null, max: n ? Number(max.toPrecision(6)) : null, mean: n ? Number((sum / n).toPrecision(6)) : null, head }
    }

    function renderText(_args, value) { return [{ type: 'text', text: JSON.stringify(value, null, 2) }] }

    function register(name, description, parameters, outputSchema, execute) {
      ctx.effect(() => harness.registerTool(ctx, harness.defineTool({
        name, description, parameters, output: { schema: outputSchema, render: renderText }, execute,
      })))
    }

    register('rave_config_validate',
      'Check a RAVE-SIM simulation directory for physical/logical validity before running: config parse, power-of-two N, FOV coverage, z-layout, cutoff-angle monotonicity, Nyquist sampling, phase-step consistency. Returns one result per check.',
      { sim_dir: { type: 'string', description: 'Absolute path of the simulation directory (contains config.yaml, computed.yaml)' } },
      { type: 'object', properties: { ok: { type: 'boolean' }, checks: { type: 'array', items: { type: 'object', additionalProperties: true } } }, additionalProperties: true },
      async function (args) {
        const simDir = String(args.sim_dir || '')
        if (!simDir) return { error: 'sim_dir is required' }
        return runValidator(['--validate', simDir])
      })

    register('rave_feasibility_check',
      'Validate a simulation directory AND check whether it can run on the current hardware: GPU memory estimate vs actual free VRAM (nvidia-smi), disk estimate for out-of-core mode, and a rough runtime estimate. Use this before rave_sim_run.',
      {
        sim_dir: { type: 'string', description: 'Absolute path of the simulation directory' },
        engine: { type: 'string', description: 'Engine to plan for: "fast-wave" (default, GPU-gated) or "big-wave" (CPU/out-of-core)' },
      },
      { type: 'object', properties: { ok: { type: 'boolean' }, feasible: { type: 'boolean' }, gpu: { type: 'object', additionalProperties: true }, est_time_min: { type: 'number' } }, additionalProperties: true },
      async function (args) {
        const simDir = String(args.sim_dir || '')
        if (!simDir) return { error: 'sim_dir is required' }
        const engine = args.engine ? String(args.engine) : 'fast-wave'
        return runValidator(['--feasibility', simDir, '--engine', engine])
      })

    register('rave_sim_run',
      'Run a RAVE-SIM fastwave simulation on the GPU. FIRST runs the feasibility gate (physics + VRAM check); refuses to start when it fails. Copies are only allowed under output/ (whitelist). Then copies the simulation directory to an isolated target, starts fastwave in the background, and returns a job_id for rave_sim_status.',
      {
        sim_dir: { type: 'string', description: 'Absolute path of the simulation directory (contains config.yaml, computed.yaml, 00000000/, ...)' },
        source_idx: { type: 'number', description: '0-based source index to simulate (default 0)' },
        fastwave_path: { type: 'string', description: 'Path to the fastwave executable (default: fast-wave/build-Release/fastwave)' },
        target_dir: { type: 'string', description: 'Optional explicit copy target directory; MUST be inside output/. Default: output/_agent_runs/<name>__agentrun_<ts>' },
        skip_check: { type: 'boolean', description: 'Set true to bypass the feasibility gate (use only when you know the sim is valid)' },
      },
      { type: 'object', properties: { job_id: { type: 'string' }, run_dir: { type: 'string' }, gpu: { type: 'object', additionalProperties: true }, pid: { type: 'number' } }, additionalProperties: true },
      async function (args) {
        const simDir = String(args.sim_dir || '')
        if (!simDir) return { error: 'sim_dir is required' }
        const idx = args.source_idx === undefined ? 0 : Number(args.source_idx)
        const fw = args.fastwave_path ? String(args.fastwave_path) : DEFAULT_FASTWAVE
        const targetDir = args.target_dir ? String(args.target_dir) : defaultTarget(simDir)
        const skip = args.skip_check === true

        if (!isUnderOutputRoot(targetDir)) {
          return { error: 'target_dir rejected: must be inside ' + OUTPUT_ROOT + ' (got ' + targetDir + ')' }
        }

        if (!skip) {
          const gate = await runValidator(['--feasibility', simDir, '--engine', 'fast-wave'])
          if (gate.error) return { error: 'feasibility gate failed to run: ' + gate.error }
          if (gate.ok !== true || gate.feasible !== true) {
            return {
              error: 'feasibility gate REJECTED the simulation before start (set skip_check=true only if you are sure)',
              feasibility: gate,
            }
          }
        }

        const gpu = await gpuInfo()
        if (!gpu.ok) return { error: 'GPU unavailable: ' + gpu.detail }
        try {
          await copySimDir(simDir, targetDir)
        } catch (e) {
          return { error: 'copy failed: ' + String(e && e.message ? e.message : e) }
        }
        const jobId = 'rv' + (++seq)
        const handle = subprocess.spawn({
          argv: [fw, '-s', String(idx), targetDir],
          cwd: targetDir,
          env: { CUDA_VISIBLE_DEVICES: '0' },
          graceMs: 5000,
          stdio: {
            stdin: 'ignore',
            stdout: { maxBytes: 8192, spill: { maxBytes: 1 << 22 } },
            stderr: { maxBytes: 8192, spill: { maxBytes: 1 << 22 } },
          },
        })
        const job = { handle, runDir: targetDir, simDir, sourceIdx: idx, startedAt: Date.now(), settled: false, exitCode: null, signal: null }
        jobs.set(jobId, job)
        handle.done.then(function (o) { job.settled = true; job.exitCode = o.exitCode; job.signal = o.signal }).catch(function () { job.settled = true; job.exitCode = null })
        return { job_id: jobId, run_dir: targetDir, source_idx: idx, gpu, pid: handle.pid, note: 'fastwave started in background; poll with rave_sim_status' }
      })

    register('rave_sim_status',
      'Query the status of a background RAVE-SIM job started by rave_sim_run: running/done/failed, elapsed time, log tail, and whether detected.npy exists.',
      { job_id: { type: 'string', description: 'Job id returned by rave_sim_run' } },
      { type: 'object', properties: { status: { type: 'string' }, elapsed_s: { type: 'number' }, exit_code: { type: 'number' }, detected: { type: 'boolean' }, log_tail: { type: 'string' } }, additionalProperties: true },
      async function (args) {
        const job = jobs.get(String(args.job_id || ''))
        if (!job) return { error: 'unknown job_id: ' + args.job_id, known_jobs: Array.from(jobs.keys()) }
        const elapsed = (Date.now() - job.startedAt) / 1000
        let detected = false
        try {
          const t = await fsService.resolve(job.runDir + '/00000000/detected.npy')
          detected = (await fsService.stat(t)) !== undefined
        } catch (e) { detected = false }
        const out = job.handle.collected.stdout ? job.handle.collected.stdout.readFrom(0).text : ''
        const err = job.handle.collected.stderr ? job.handle.collected.stderr.readFrom(0).text : ''
        const tail = (out + err).replace(/\s+$/, '').split('\n').slice(-25).join('\n')
        const status = job.settled ? (job.exitCode === 0 ? 'done' : 'failed') : 'running'
        return { job_id: args.job_id, status, elapsed_s: Number(elapsed.toFixed(1)), exit_code: job.exitCode === null ? -1 : job.exitCode, signal: job.signal === null ? 'none' : job.signal, detected, run_dir: job.runDir, log_tail: tail }
      })

    register('rave_result_summary',
      'Read a .npy result file (e.g. detected.npy) and return its shape, dtype, min/max/mean and first values — safe for large arrays since only summary stats are returned.',
      { path: { type: 'string', description: 'Absolute path to a .npy file' } },
      { type: 'object', properties: { shape: { type: 'array', items: { type: 'number' } }, dtype: { type: 'string' }, min: { type: 'number' }, max: { type: 'number' }, mean: { type: 'number' } }, additionalProperties: true },
      async function (args) {
        try { return await npySummary(String(args.path)) } catch (e) { return { error: String(e && e.message ? e.message : e) } }
      })

    register('rave_result_plot',
      'Generate a PNG plot of a result .npy file (e.g. detected.npy) — 1D profile or 2D image, auto-detected. PNG is saved under output/_agent_runs/plots/, rendered inline in the conversation card, and a markdown-embeddable http `url` is returned. Embed the image in your reply with ![title](url).',
      {
        path: { type: 'string', description: 'Absolute path to a .npy file (e.g. .../00000000/detected.npy)' },
        sim_dir: { type: 'string', description: 'Alternative: simulation directory; detected.npy under 00000000/ is plotted automatically' },
      },
      { type: 'object', properties: { png_path: { type: 'string' }, url: { type: 'string' }, shape: { type: 'array', items: { type: 'number' } }, kind: { type: 'string' }, min: { type: 'number' }, max: { type: 'number' }, mean: { type: 'number' } }, additionalProperties: true },
      async function (args) {
        const p = String(args.path || '')
        const sd = String(args.sim_dir || '')
        const npyPath = p || (sd ? resolveDetected(sd) : '')
        if (!npyPath) return { error: 'path or sim_dir required' }
        return attachPlotUrl(await makePlot(npyPath, sd || undefined))
      })

    register('rave_grid_plot',
      'Render the sample density/material grid(s) referenced by a simulation directory (config.yaml elements[].grid_path, e.g. the shockwave grid) into PNG(s) under output/_agent_runs/plots/. Returns per-grid png_path, a markdown-embeddable http `url`, shape and numeric stats (grid index range + density range in g/cm³). Embed the image in your reply with ![title](url).',
      {
        sim_dir: { type: 'string', description: 'Absolute path of the simulation directory (contains config.yaml)' },
        grid: { type: 'string', description: 'Optional explicit path to a grid .npy; default: all grids from config.yaml' },
      },
      { type: 'object', properties: { ok: { type: 'boolean' }, grids: { type: 'array', items: { type: 'object', additionalProperties: true } } }, additionalProperties: true },
      async function (args) {
        const simDir = String(args.sim_dir || '')
        if (!simDir) return { error: 'sim_dir is required' }
        await ensurePlotServer()
        return makeGridPlot(simDir, args.grid ? String(args.grid) : undefined)
      })

    // RPC for the in-conversation viewer (Client half calls this)
    ctx.effect(() => harness.handle('rave-plot-png', async (args) => {
      const p = args && args.path ? String(args.path) : ''
      const sd = args && args.sim_dir ? String(args.sim_dir) : ''
      const npyPath = p || (sd ? resolveDetected(sd) : '')
      const res = (args && args.grid)
        ? await makeGridPlot(sd, args.grid ? String(args.grid) : undefined)
        : await makePlot(npyPath, sd || undefined)
      if (res.error || !res.png_path) return { error: res.error || 'no png produced' }
      try {
        const u8 = await fsService.readBytes(await fsService.resolve(res.png_path), undefined, 8 * 1024 * 1024)
        return { dataUrl: 'data:image/png;base64,' + bytesToBase64(u8), png_path: res.png_path, shape: res.shape, kind: res.kind }
      } catch (e) {
        return { error: String(e && e.message ? e.message : e) }
      }
    }))

    ctx.effect(function () {
      return function () {
        for (const job of jobs.values()) { try { job.handle.terminate() } catch (e) { /* ignore */ } }
        jobs.clear()
      }
    })

    console.log('[rave-sim] plugin v10 loaded: 7 tools + plot RPC (validate, feasibility, run, status, summary, plot, grid plot)')
  },
}
