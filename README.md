# Tomte

Moves textures from one UV layout to another on the same mesh. Colour is
resampled in linear light; tangent-space normal maps are re-oriented, so
rotated, mirrored or rescaled islands come out right.

Named after the tomte, the Swedish house gnome who quietly does the chores
while everyone sleeps.

> **AI disclosure:** this is an AI-assisted course project. Much of the code,
> tests and documentation was written with Claude Code (Anthropic's Claude),
> directed by the author. You're free to dislike that, and equally free not to
> use this tool.

## Install

Needs Python 3.10 or later.

- **Windows:** double-click **`Tomte.bat`**. The first time, it offers to
  install the libraries it needs.
- **Anywhere:** in this folder, run `pip install -r requirements.txt`, then
  `python -m uvtransfer.gui`. Or run `pip install ".[gui,precision]"` to get a
  `tomte` command (opens the app) and a `uvtransfer` command (command line)
  that work from any folder.

## Desktop app

1. Pick the **source FBX** (the UVs the textures were made for) and the
   **target FBX** (the same mesh with new UVs). The window checks that they match.
2. Drop in textures. Each is typed Colour, Normal map or Data from its file
   name; change it in the dropdown if the guess is wrong.
3. Press **Transfer**. Results go to `transferred/` next to the target FBX
   unless you choose another folder.

Edited a mesh or texture since? Press **Reload** (F5) to read everything
again from disk; finished textures whose files changed are marked
"File changed".

### Setups for repeat bakes

**File > Save setup** (Ctrl+S) stores both meshes, the textures and their
types, and the output settings in a small `.tomte` file. **File > Open
setup** (Ctrl+O), dropping the file on the window, or dropping it on
`Tomte.bat` brings it all back.

**File > Copy command line** saves the setup and copies a command that runs
it without opening the app, ready to paste into a terminal, a script or a
scheduled task:

    python "C:\Tools\Tomte\tomte_cli.py" --setup "D:\Project\Body.tomte"

Setup files are plain JSON with paths relative to the file, so a project
folder can be moved or shared with the setup inside it.

## Command line

    python tomte_cli.py SOURCE.fbx TARGET.fbx tex1.png [tex2.png ...] [-o DIR]
        [--size N] [--ss N] [--pad N] [--normal-convention auto|dx|gl]
        [--as NAME=normal|srgb|linear] [--uv-set I] [--16bit]

    python tomte_cli.py --setup Body.tomte [options]
    python tomte_cli.py SOURCE.fbx TARGET.fbx tex1.png ... --save-setup Body.tomte

`tomte_cli.py` works from any folder. Options given with `--setup` override
the file's; `--save-setup` writes a setup instead of transferring. From this
folder, `python -m uvtransfer` does the same. Run with `--help` for details.

## Try it

`samples/` has a sphere with two UV layouts and a colour and normal map made
for the first. The second layout rotates, mirrors and shrinks the strips; the
letters show whether each one came out the right way round.

    python tomte_cli.py --setup samples/Sphere.tomte

Or open `samples/Sphere.tomte` in the app. Results land in `samples/transferred/`.

## Python

```python
from uvtransfer import Options, Transfer

xfer = Transfer.from_fbx("samples/Sphere_OriginalUVs.fbx", "samples/Sphere_NewUVs.fbx")
print(xfer.summary(), xfer.warnings)
result = xfer.transfer_file("samples/Sphere_Normal.png", "out", options=Options(pad=16))
```

## Good to know

- **Export triangulated meshes.** Quads can be split two ways and tools don't
  agree; triangulating before export (and baking/rendering that same mesh)
  keeps tangents consistent everywhere.
- Tangent frames follow MikkTSpace closely but not exactly (see
  `uvtransfer/tangents.py` for the measured differences).
- UDIM tiles aren't supported: UVs outside 0 to 1 wrap onto one texture.
- 16-bit colour PNG/TIFF input needs `opencv-python-headless` for full
  precision; without it the app warns and reads 8 bits.

## Feedback

Bug reports and suggestions are welcome as GitHub issues. Pull requests
aren't accepted.

## Licence

Copyright (C) 2026 feanix

Tomte is free software: you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version. See `COPYING.LESSER` and `COPYING`.

It is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE.
