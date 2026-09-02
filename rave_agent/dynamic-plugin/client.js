// RAVE-SIM result viewer: inline plot/grid cards in the conversation
// (rave_result_plot + rave_grid_plot)
return {
  apply(ctx) {
    const slots = ctx.get('slots')
    if (slots === undefined) return

    function RavePlotView(props) {
      const block = props.block
      const done = block && 'kind' in block
      const argsRaw = done ? (block.call && block.call.argsRaw) : (block ? block.argsRaw : '')
      let path = null
      let simDir = null
      let grid = false
      try {
        const a = JSON.parse(argsRaw || '{}')
        path = a.path || null
        simDir = a.sim_dir || null
        grid = !!a.grid
      } catch (e) { /* ignore */ }
      const [img, setImg] = React.useState(null)
      const [err, setErr] = React.useState(null)
      React.useEffect(function () {
        if (!path && !simDir) return
        let alive = true
        const rpcArgs = {}
        if (path) rpcArgs.path = path
        if (simDir) rpcArgs.sim_dir = simDir
        if (grid) rpcArgs.grid = true
        host.call('rave-plot-png', rpcArgs)
          .then(function (r) {
            if (!alive) return
            if (r && r.dataUrl) setImg(r.dataUrl)
            else setErr(r && r.error ? r.error : 'no plot data')
          })
          .catch(function (e) { if (alive) setErr(String(e)) })
        return function () { alive = false }
      }, [path, simDir, grid])
      if (!path && !simDir) {
        return React.createElement('div', null, 'rave_result_plot / rave_grid_plot: pass path or sim_dir')
      }
      return React.createElement('div', { style: { padding: '8px 0' } },
        err ? React.createElement('div', { style: { color: '#d64' } }, 'plot error: ' + err) : null,
        img
          ? React.createElement('img', { src: img, alt: 'plot', style: { maxWidth: '100%', borderRadius: 8, display: 'block' } })
          : React.createElement('div', null, 'loading plot…'))
    }

    slots.inject('tool.call.toolview', () => slots.register(
      { name: 'tool.call.toolview', key: 'rave_result_plot' },
      (props) => RavePlotView(props),
    ))
    slots.inject('tool.call.toolview', () => slots.register(
      { name: 'tool.call.toolview', key: 'rave_grid_plot' },
      (props) => RavePlotView(props),
    ))
    console.log('[rave-sim] client viewer registered for rave_result_plot + rave_grid_plot')
  },
}
