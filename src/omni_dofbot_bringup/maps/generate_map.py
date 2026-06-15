#!/usr/bin/env python3
"""
generate_arena_map.py — v2

Genera el mapa estático PGM + YAML para Nav2.
Cambios: O1 en (0.00, +0.25), O2 en (0.00, -0.30).

Uso:
  python3 generate_arena_map.py
  cp arena_map.pgm arena_map.yaml src/omni_dofbot_bringup/maps/
"""

import numpy as np

RESOLUTION = 0.02
ARENA_W    = 1.22
ARENA_H    = 2.42
WALL_T     = 0.03
ORIGIN_X   = -ARENA_W / 2.0   # -0.61
ORIGIN_Y   = -ARENA_H / 2.0   # -1.21
MAP_W      = int(ARENA_W / RESOLUTION)   # 61
MAP_H      = int(ARENA_H / RESOLUTION)   # 121

FREE     = 254
OCCUPIED = 0

# (x_center, y_center, width, height)
OBSTACLES = [
    (0.00,  0.25, 0.15, 0.15),   # O1 — centro norte
    (0.00, -0.30, 0.15, 0.15),   # O2 — centro sur
]


def world_to_pixel(x, y):
    col = int((x - ORIGIN_X) / RESOLUTION)
    row = int((y - ORIGIN_Y) / RESOLUTION)
    return col, row


def draw_rect(grid, xc, yc, w, h, value, margin=0.0):
    c0, r0 = world_to_pixel(xc - w/2 - margin, yc - h/2 - margin)
    c1, r1 = world_to_pixel(xc + w/2 + margin, yc + h/2 + margin)
    c0 = max(0, min(MAP_W-1, c0)); c1 = max(0, min(MAP_W-1, c1))
    r0 = max(0, min(MAP_H-1, r0)); r1 = max(0, min(MAP_H-1, r1))
    grid[r0:r1+1, c0:c1+1] = value


def main():
    print('Generando mapa v2 — cajas centradas en x=0')
    grid = np.full((MAP_H, MAP_W), FREE, dtype=np.uint8)

    # Paredes
    draw_rect(grid,  0,  ARENA_H/2 - WALL_T/2, ARENA_W, WALL_T, OCCUPIED)
    draw_rect(grid,  0, -ARENA_H/2 + WALL_T/2, ARENA_W, WALL_T, OCCUPIED)
    draw_rect(grid,  ARENA_W/2 - WALL_T/2, 0, WALL_T, ARENA_H, OCCUPIED)
    draw_rect(grid, -ARENA_W/2 + WALL_T/2, 0, WALL_T, ARENA_H, OCCUPIED)

    # Obstáculos con margen de 1 px
    for (xc, yc, w, h) in OBSTACLES:
        draw_rect(grid, xc, yc, w, h, OCCUPIED, margin=RESOLUTION)

    # Guardar PGM
    flipped = np.flipud(grid)
    with open('arena_map.pgm', 'wb') as f:
        f.write(f'P5\n{MAP_W} {MAP_H}\n255\n'.encode())
        f.write(flipped.tobytes())

    # Guardar YAML
    with open('arena_map.yaml', 'w') as f:
        f.write(f"""image: arena_map.pgm
resolution: {RESOLUTION}
origin: [{ORIGIN_X}, {ORIGIN_Y}, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.25
""")

    free = np.sum(grid == FREE)
    occ  = np.sum(grid == OCCUPIED)
    print(f'  Mapa: {MAP_W}×{MAP_H} px  libre={free}  ocupado={occ}')
    print('Listo. Copia los archivos:')
    print('  cp arena_map.pgm arena_map.yaml src/omni_dofbot_bringup/maps/')


if __name__ == '__main__':
    main()