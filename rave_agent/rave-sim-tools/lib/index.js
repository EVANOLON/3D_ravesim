// RAVE-SIM tools — persisted Cordis plugin (static npm-package form).
//
// This persisted form is now a BOOTSTRAP: it registers one tool,
// `rave_plugin_activate`, which reads the persisted dynamic-plugin sources
// (rave_agent/dynamic-plugin/host.js + client.js) and activates them through
// the `dynamicCordisRunner` service — the same API the cordis_define/cordis_run
// tools use, without needing the cordis toolset in this preset.
//
// The dynamic plugin then owns all 6 business tools (validate / feasibility /
// run / status / summary / plot) plus the in-conversation inline plot viewer.
// Keeping the business tools only in the dynamic plugin avoids duplicate
// registrations of the same tool names.
//
// Workflow after DSH restart:
//   1. Start a session on the RAVE-SIM preset.
//   2. The agent (or the user) calls `rave_plugin_activate`.
//   3. Approve the Run card once (Client half).
//   4. All 6 tools + inline viewer are live for that session.

const HOST_SRC = '/mnt/d/rave-sim-main/rave-sim-main/rave_agent/dynamic-plugin/host.js'
const CLIENT_SRC = '/mnt/d/rave-sim-main/rave-sim-main/rave_agent/dynamic-plugin/client.js'
const PLOT_SERVER_SCRIPT = '/mnt/d/rave-sim-main/rave-sim-main/rave_agent/plot_server.py'
const PYTHON = '/home/taylor/anaconda3/envs/rave-sim/bin/python'

export default {
  name: 'rave-sim-tools',
  inject: ['fs', 'tools', 'subprocess', 'dynamicCordisRunner'],
  apply(ctx) {
    const fsService = ctx.fs
    const tools = ctx.tools
    const runner = ctx.get('dynamicCordisRunner')

    function renderText(_args, value) {
      return [{ type: 'text', text: JSON.stringify(value, null, 2) }]
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

    // ── rave_plugin_activate ──────────────────────────────────────────────
    tools.register({
      name: 'rave_plugin_activate',
      description: 'Activate the RAVE-SIM dynamic plugin (6 GPU tools + inline plot viewer) for this session. Reads rave_agent/dynamic-plugin/host.js and client.js, defines a new plugin via dynamicCordisRunner and starts it. If it needs browser approval you will see a Run card; approve it once. Safe to call repeatedly — reuses an existing plugin.',
      parameters: {},
      output: {
        schema: { type: 'object', properties: { ok: { type: 'boolean' }, plugin_id: { type: 'string' }, package_id: { type: 'string' }, run_status: { type: 'string' } }, additionalProperties: true },
        render: renderText,
      },
      async execute(args, exec) {
        const agent = exec && exec.agent ? exec.agent : null
        const rnr = ctx.get('dynamicCordisRunner')
        if (!agent) return { error: 'no agent context available' }
        if (rnr === undefined) return { error: 'dynamicCordisRunner unavailable (host composition missing this service?)' }
        try {
          const hostSrc = await fsService.readText(await fsService.resolve(HOST_SRC))
          const clientSrc = await fsService.readText(await fsService.resolve(CLIENT_SRC))
          const existing = rnr.listPlugins(agent).find(function (p) { return p.name === 'rave-sim-gpu-full' })
          let pluginId, packageId
          if (existing) {
            pluginId = existing.pluginId
            packageId = existing.currentPackageId || existing.packageId
          } else {
            const sessionId = (agent.session && agent.session.id) ? agent.session.id : ''
            const receipt = rnr.define({
              sessionId,
              plugin: { kind: 'new', idPrefix: 'rave' },
              name: 'rave-sim-gpu-full',
              purpose: 'RAVE-SIM GPU tools + inline plot viewer (auto-activated by bootstrap)',
              code: { host: hostSrc, client: clientSrc },
            })
            pluginId = receipt.pluginId
            packageId = receipt.packageId
          }
          const res = await rnr.run(agent, pluginId, packageId, 'run')
          const status = res && (res.status || res.kind) ? String(res.status || res.kind) : 'started'
          return {
            ok: true,
            plugin_id: pluginId,
            package_id: packageId,
            run_status: status,
            note: status === 'awaiting-approval'
              ? 'approve the Run card in the UI; tools appear afterwards'
              : 'dynamic plugin activating; tools appear from the next step',
          }
        } catch (e) {
          return { error: String(e && e.message ? e.message : e) }
        }
      },
    })

    // ── rave_plugin_status ────────────────────────────────────────────────
    tools.register({
      name: 'rave_plugin_status',
      description: 'Report whether the RAVE-SIM dynamic plugin is active for this session and list its packages.',
      parameters: {},
      output: {
        schema: { type: 'object', properties: { active: { type: 'boolean' }, plugins: { type: 'array', items: { type: 'object', additionalProperties: true } } }, additionalProperties: true },
        render: renderText,
      },
      async execute(args, exec) {
        const agent = exec && exec.agent ? exec.agent : null
        const rnr = ctx.get('dynamicCordisRunner')
        if (!agent || rnr === undefined) return { active: false, error: 'runner or agent unavailable' }
        try {
          const plugins = rnr.listPlugins(agent)
          const active = plugins.some(function (p) { return p.name === 'rave-sim-gpu-full' && p.activeRun })
          return {
            active,
            plugins: plugins.map(function (p) {
              return { pluginId: p.pluginId, name: p.name, currentPackageId: p.currentPackageId || null, hasRun: !!p.activeRun }
            }),
          }
        } catch (e) {
          return { active: false, error: String(e && e.message ? e.message : e) }
        }
      },
    })

    // ── rave_plot_server ──────────────────────────────────────────────────
    // Available from the first turn (no plugin activation needed): keeps the
    // inline-plot HTTP server up so result/grid PNGs can be embedded in the
    // conversation via markdown ![title](http://127.0.0.1:8811/<file>.png).
    tools.register({
      name: 'rave_plot_server',
      description: 'Ensure the RAVE-SIM inline-plot HTTP server is running at http://127.0.0.1:8811 (serves output/_agent_runs/plots). Call once at session start (action ensure). The returned url lets you embed result/grid PNGs directly in the conversation via markdown ![title](url). Also lists the PNGs currently available.',
      parameters: {
        action: {
          type: 'string', enum: ['ensure', 'status', 'stop'],
          description: 'ensure (default): start the server if it is not running; status: report only; stop: shut the server down',
        },
      },
      output: {
        schema: { type: 'object', properties: { ok: { type: 'boolean' }, up: { type: 'boolean' }, url: { type: 'string' }, pngs: { type: 'array', items: { type: 'string' } } }, additionalProperties: true },
        render: renderText,
      },
      async execute(args) {
        const action = args && args.action ? String(args.action) : 'ensure'
        const r = await runScript([PYTHON, PLOT_SERVER_SCRIPT, action], 8192)
        if (r.error) return { error: r.error }
        const text = r.out.trim()
        if (text) {
          try { return JSON.parse(text) } catch (e) { /* fall through */ }
        }
        return { error: 'plot server script exited ' + r.outcome.exitCode + ': ' + ((r.err || text).slice(0, 500) || '(no output)') }
      },
    })

    console.log('[rave-sim-tools] bootstrap registered: rave_plugin_activate, rave_plugin_status, rave_plot_server')
  },
}
