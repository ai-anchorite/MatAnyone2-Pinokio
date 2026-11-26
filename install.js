module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -e .",
          "uv pip install -r requirements.txt"
        ]
      }
    },

    {
      method: "script.start",
      params: {
        uri: "{{platform === 'win32' && gpu === 'amd' ? 'torch_amd.js' : 'torch.js'}}",
        params: {
          venv: "env",
          path: "app",
        }
      }
    },

    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install \"pillow>=9.5,<12.0\""
        ]
      }
    },

    {
      method: "notify",
      params: {
        html: "Installation complete! Click Start to run MatAnyone2-Pinokio!."
      }
    }
  ]
}
