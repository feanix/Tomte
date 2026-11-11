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

### Several parts, each with its own textures

Sometimes the textures come in sets: a head, teeth and eyes, each painted
in its own 0 to 1 UV space, now packed together into one new layout. Tell
Tomte which set each part belongs to by moving the parts into separate
UDIM tiles in the **source** FBX: the head stays in 1001 (UVs 0 to 1), the
teeth move one step right to 1002 (UVs 1 to 2), the eyes to 1003, and so on.
The target FBX keeps everything in 0 to 1.

The texture list then shows one row per output and a column per tile:

1. Drop each part's textures onto its tile's column. Textures of the same
   kind (colour, normal, or the same last word such as "ORM") line up in
   the same row. Files named the UDIM way (`Body_Color.1002.png`) find
   their own tile.
2. Double-click an output's name to rename it.
3. **Fill** is the colour for parts whose tile has no texture in that row
   (or **Empty**, so edge padding covers them). Normal maps are always
   flat there.

Each tile's normal map format is detected on its own; the result uses the
format of the largest part and converts the rest to match.

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

For a source FBX with parts in UDIM tiles, textures named
`Body_Color.1001.png`, `Body_Color.1002.png` ... are put together into
`Body_Color.png`. For textures named otherwise, give each output with
`--map` (after the FBX files):

    --map Body_Color.png 1001=Head_Color.png 1002=Teeth_Color.png
    --map Body_ORM.png 1001=Head_ORM.png fill=1,0.5,0

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
- UDIM tiles work in the source mesh (one texture per tile, see above). The
  target is a single 0 to 1 texture: target UVs outside 0 to 1 wrap onto it.
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
