### Docker manual build

NOTE: Needs approx. 45GB of space. Could be optimized (see dockerfiles).

```bash
docker build -f dockerfiles/torch_geometric_base.dockerfile --target torch_geometric_base -t torch_geometric_base:local .
```

```bash
docker build -f dockerfiles/dano-pt.dockerfile -t dano-pt:local .
```

## VSCode

```json
{
  "name": "dano-pt",

  "image": "dano-pt:local",

  "workspaceFolder": "/workspace",
  "runArgs": [
    "--gpus", "all",
    "-v", "${localWorkspaceFolder}:/workspace:cached"
  ],
  "overrideCommand": true,

  "customizations": {
    "vscode": {
      "settings": {
        "terminal.integrated.shell.linux": "/bin/bash"
      },
      "extensions": [
        "ms-python.python",
        "njpwerner.autodocstring",
        "eamodio.gitlens",
        "ms-azuretools.vscode-docker"  // Dockerfile syntax and IntelliSense
      ]
    }
  }
}
```
