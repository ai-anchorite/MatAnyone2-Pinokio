// Windows AMD torch/ROCm installer — ALL platform-specific GPU stack logic
// for win32+amd lives here (install.js delegates here after venv creation;
// update.js never touches torch, so whatever install.js leaves is what runs).
//
// Torch comes from TheRock's unified nightly index (ROCm/TheRock RELEASES.md):
//   https://nightly.repo.amd.com/rocm/whl-next/
// Pinned to the 2.14.0 STABLE line (unpinned --pre jumps to alphas; ==2.14.0
// matches the newest dated +rocm local build via PEP 440). TheRock builds
// cp310-cp314, so this file is python-version agnostic. The ROCm ecosystem
// moves weekly — before bumping anything re-check the index layout
// (RELEASES.md) and the triton-windows pin below.
//
// Detection: branches match `gpu_target` (Pinokio's arch var: "gfx1201" etc.)
// with the GPU model name as fallback. Unknown identity → loud warning branch
// that installs NOTHING (2026-09-08 incident: a blind fallback once
// force-installed torch-directml over a working ROCm env).
//
// Every torch install uses -U (else uv no-ops on any installed torch) and
// uninstalls torch-directml first (self-heal for clobbered envs). The
// retired 0xDELUXA accelerator wheels are gone — nothing in iw3 needs them.
module.exports = {
  run: [
    // RDNA4 — RX 9070 / 9070 GRE / 9070 XT, AI PRO R9700 (gfx1201) — primary path
    // (must run after gfx1200 so 9060-class cards are caught first).
    // rocm-sdk-devel (5.2GB HIP toolchain) was dropped 2026-10-16: torch's
    // runtime comes from the rocm[libraries] metapackage it requires itself;
    // nothing here compiles HIP (verified live — convs fine without it).
    // The uninstall self-heals senzu-era installs that still carry it.
    {
      "when": "{{platform === 'win32' && gpu === 'amd' && /gfx1201|9\\d{3}/i.test(gpu_target || kernel.gpu_model || gpu_model || '')}}",
      method: "shell.run",
      params: {
        "bluefairy": "off",
        "env": { "UV_SKIP_WHEEL_FILENAME_CHECK": "1" },
        "venv_python": "{{args && args.venv_python ? args.venv_python : null}}",
        "venv": "{{args && args.venv ? args.venv : null}}",
        "path": "{{args && args.path ? args.path : '.'}}",
        "message": [
          "echo [torch_amd] RDNA4 gfx1201 branch",
          "uv pip uninstall torch-directml rocm-sdk-devel",
          "uv pip install -U --pre \"torch[device-gfx1201]==2.14.0\" \"torchvision[device-gfx1201]\" torchaudio --index-url https://nightly.repo.amd.com/rocm/whl-next/"
        ]
      }
    },
    // RDNA4 — RX 9060 / 9060 XT (gfx1200)
    {
      "when": "{{platform === 'win32' && gpu === 'amd' && /gfx1200|9060/i.test(gpu_target || kernel.gpu_model || gpu_model || '')}}",
      "method": "shell.run",
      "params": {
        "bluefairy": "off",
        "env": { "UV_SKIP_WHEEL_FILENAME_CHECK": "1" },
        "venv_python": "{{args && args.venv_python ? args.venv_python : null}}",
        "venv": "{{args && args.venv ? args.venv : null}}",
        "path": "{{args && args.path ? args.path : '.'}}",
        "message": [
          "echo [torch_amd] RDNA4 gfx1200 branch",
          "uv pip uninstall torch-directml",
          "uv pip install -U --pre \"torch[device-gfx1200]==2.14.0\" \"torchvision[device-gfx1200]\" torchaudio --index-url https://nightly.repo.amd.com/rocm/whl-next/"
        ]
      }
    },
    // RDNA4 — gfx1250 (future AI PRO parts)
    {
      "when": "{{platform === 'win32' && gpu === 'amd' && /gfx1250/i.test(gpu_target || '')}}",
      "method": "shell.run",
      "params": {
        "bluefairy": "off",
        "env": { "UV_SKIP_WHEEL_FILENAME_CHECK": "1" },
        "venv_python": "{{args && args.venv_python ? args.venv_python : null}}",
        "venv": "{{args && args.venv ? args.venv : null}}",
        "path": "{{args && args.path ? args.path : '.'}}",
        "message": [
          "echo [torch_amd] RDNA4 gfx1250 branch",
          "uv pip uninstall torch-directml",
          "uv pip install -U --pre \"torch[device-gfx1250]==2.14.0\" \"torchvision[device-gfx1250]\" torchaudio --index-url https://nightly.repo.amd.com/rocm/whl-next/"
        ]
      }
    },
    // RDNA3.5 — Strix Halo / Ryzen AI Max / 8060S (gfx1151)
    {
      "when": "{{platform === 'win32' && gpu === 'amd' && /gfx1151|8060|Ryzen AI Max/i.test(gpu_target || kernel.gpu_model || gpu_model || '')}}",
      "method": "shell.run",
      "params": {
        "bluefairy": "off",
        "env": { "UV_SKIP_WHEEL_FILENAME_CHECK": "1" },
        "venv_python": "{{args && args.venv_python ? args.venv_python : null}}",
        "venv": "{{args && args.venv ? args.venv : null}}",
        "path": "{{args && args.path ? args.path : '.'}}",
        "message": [
          "echo [torch_amd] Strix Halo gfx1151 branch",
          "uv pip uninstall torch-directml",
          "uv pip install -U --pre \"torch[device-gfx1151]==2.14.0\" \"torchvision[device-gfx1151]\" torchaudio --index-url https://nightly.repo.amd.com/rocm/whl-next/"
        ]
      }
    },
    // RDNA3.5 — Strix Point 880M/890M (gfx1150)
    {
      "when": "{{platform === 'win32' && gpu === 'amd' && /gfx1150|8[89]0M/i.test(gpu_target || kernel.gpu_model || gpu_model || '')}}",
      "method": "shell.run",
      "params": {
        "bluefairy": "off",
        "env": { "UV_SKIP_WHEEL_FILENAME_CHECK": "1" },
        "venv_python": "{{args && args.venv_python ? args.venv_python : null}}",
        "venv": "{{args && args.venv ? args.venv : null}}",
        "path": "{{args && args.path ? args.path : '.'}}",
        "message": [
          "echo [torch_amd] Strix Point gfx1150 branch",
          "uv pip uninstall torch-directml",
          "uv pip install -U --pre \"torch[device-gfx1150]==2.14.0\" \"torchvision[device-gfx1150]\" torchaudio --index-url https://nightly.repo.amd.com/rocm/whl-next/"
        ]
      }
    },
    // RDNA3.5 — other gfx115x (gfx1152/1153) via combined extra
    {
      "when": "{{platform === 'win32' && gpu === 'amd' && /gfx115[23]/i.test(gpu_target || '')}}",
      "method": "shell.run",
      "params": {
        "bluefairy": "off",
        "env": { "UV_SKIP_WHEEL_FILENAME_CHECK": "1" },
        "venv_python": "{{args && args.venv_python ? args.venv_python : null}}",
        "venv": "{{args && args.venv ? args.venv : null}}",
        "path": "{{args && args.path ? args.path : '.'}}",
        "message": [
          "echo [torch_amd] gfx115x branch",
          "uv pip uninstall torch-directml",
          "uv pip install -U --pre \"torch[device-gfx115x]==2.14.0\" \"torchvision[device-gfx115x]\" torchaudio --index-url https://nightly.repo.amd.com/rocm/whl-next/"
        ]
      }
    },
    // RDNA3 — RX 7000 dGPUs + 7x0M iGPUs (gfx110x combined extra)
    {
      "when": "{{platform === 'win32' && gpu === 'amd' && /gfx110\\d|7\\d{3}|7\\d{2}M/i.test(gpu_target || kernel.gpu_model || gpu_model || '')}}",
      "method": "shell.run",
      "params": {
        "bluefairy": "off",
        "env": { "UV_SKIP_WHEEL_FILENAME_CHECK": "1" },
        "venv_python": "{{args && args.venv_python ? args.venv_python : null}}",
        "venv": "{{args && args.venv ? args.venv : null}}",
        "path": "{{args && args.path ? args.path : '.'}}",
        "message": [
          "echo [torch_amd] RDNA3 gfx110x branch",
          "uv pip uninstall torch-directml",
          "uv pip install -U --pre \"torch[device-gfx110x]==2.14.0\" \"torchvision[device-gfx110x]\" torchaudio --index-url https://nightly.repo.amd.com/rocm/whl-next/"
        ]
      }
    },
    // RDNA2 — RX 6000 series (gfx1030/1031/1032/1035 via multi-extra)
    {
      "when": "{{platform === 'win32' && gpu === 'amd' && /gfx103\\d|6\\d{3}/i.test(gpu_target || kernel.gpu_model || gpu_model || '')}}",
      "method": "shell.run",
      "params": {
        "bluefairy": "off",
        "env": { "UV_SKIP_WHEEL_FILENAME_CHECK": "1" },
        "venv_python": "{{args && args.venv_python ? args.venv_python : null}}",
        "venv": "{{args && args.venv ? args.venv : null}}",
        "path": "{{args && args.path ? args.path : '.'}}",
        "message": [
          "echo [torch_amd] RDNA2 branch",
          "uv pip uninstall torch-directml",
          "uv pip install -U --pre \"torch[device-gfx1030,device-gfx1031,device-gfx1032,device-gfx1035]==2.14.0\" \"torchvision[device-gfx1030,device-gfx1031,device-gfx1032,device-gfx1035]\" torchaudio --index-url https://nightly.repo.amd.com/rocm/whl-next/"
        ]
      }
    },
    // RDNA1 — RX 5000 series (gfx1010/1012 via multi-extra) — experimental
    {
      "when": "{{platform === 'win32' && gpu === 'amd' && /gfx101\\d|5\\d{3}/i.test(gpu_target || kernel.gpu_model || gpu_model || '')}}",
      "method": "shell.run",
      "params": {
        "bluefairy": "off",
        "env": { "UV_SKIP_WHEEL_FILENAME_CHECK": "1" },
        "venv_python": "{{args && args.venv_python ? args.venv_python : null}}",
        "venv": "{{args && args.venv ? args.venv : null}}",
        "path": "{{args && args.path ? args.path : '.'}}",
        "message": [
          "echo [torch_amd] RDNA1 branch, experimental",
          "uv pip uninstall torch-directml",
          "uv pip install -U --pre \"torch[device-gfx1010,device-gfx1012]==2.14.0\" \"torchvision[device-gfx1010,device-gfx1012]\" torchaudio --index-url https://nightly.repo.amd.com/rocm/whl-next/"
        ]
      }
    },
    // DirectML fallback — ONLY when the GPU is positively identified as
    // unsupported by the ROCm wheels above (Vega/GCN/etc.). Never reached on
    // detection failure — see the unknown-identity branch below.
    {
      "when": "{{platform === 'win32' && gpu === 'amd' && (gpu_target || kernel.gpu_model || gpu_model) && !/gfx1/i.test('' + (gpu_target || kernel.gpu_model || gpu_model))}}",
      "method": "shell.run",
      "params": {
        "bluefairy": "off",
        "venv_python": "{{args && args.venv_python ? args.venv_python : null}}",
        "venv": "{{args && args.venv ? args.venv : null}}",
        "path": "{{args && args.path ? args.path : '.'}}",
        "message": [
          "echo [torch_amd] DirectML fallback — GPU identified as unsupported by ROCm wheels",
          "uv pip install torch-directml torchaudio torchvision numpy==1.26.4 --force-reinstall"
        ]
      },
      "next": null
    },
    // Unknown identity — GPU detection unavailable (e.g. GPU busy during the
    // run). Fires ONLY when every identity var is falsy; ROCm branches above
    // fall through this step (identity present → when false) to reach the
    // accelerator stack. Install NOTHING: a blind PyPI install would clobber
    // any existing ROCm environment (the 2026-09-08 incident). Advise and stop.
    {
      "when": "{{platform === 'win32' && gpu === 'amd' && !(gpu_target || kernel.gpu_model || gpu_model)}}",
      "method": "shell.run",
      "params": {
        "venv_python": "{{args && args.venv_python ? args.venv_python : null}}",
        "venv": "{{args && args.venv ? args.venv : null}}",
        "path": "{{args && args.path ? args.path : '.'}}",
        "message": [
          "echo [torch_amd] WARNING: AMD GPU detected but model/arch unavailable — skipping torch install to protect the existing environment.",
          "echo [torch_amd] Restart Pinokio — or re-run Update with the GPU idle — so GPU detection can run, then try again.",
          "echo [torch_amd] If it persists, check gpu detection in Pinokio settings."
        ]
      },
      "next": null
    },

    // ── Triton ──────────────────────────────────────────────────────────────
    // Reached only through a ROCm branch above. triton-windows powers
    // torch.compile on Windows AMD — the one accelerator dependency that
    // still lives outside torch itself.
    {
      "when": "{{platform === 'win32' && gpu === 'amd'}}",
      "method": "shell.run",
      "params": {
        "bluefairy": "off",
        "venv_python": "{{args && args.venv_python ? args.venv_python : null}}",
        "venv": "{{args && args.venv ? args.venv : null}}",
        "path": "{{args && args.path ? args.path : '.'}}",
        "message": [
          "uv pip install triton-windows==3.7.1.post27"
        ]
      }
    },

    // Sanity check — torch + HIP + GPU visible
    {
      "when": "{{platform === 'win32' && gpu === 'amd'}}",
      "method": "shell.run",
      "params": {
        "venv": "{{args && args.venv ? args.venv : null}}",
        "path": "{{args && args.path ? args.path : '.'}}",
        "message": "python -c \"import torch; print(f'PyTorch: {torch.__version__}'); print(f'ROCm avail: {torch.cuda.is_available()}'); print(f'HIP: {torch.version.hip if torch.cuda.is_available() else \\\"N/A\\\"}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \\\"N/A\\\"}')\""
      }
    }
  ]
}
