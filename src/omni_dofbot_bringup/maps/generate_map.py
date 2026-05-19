#!/usr/bin/env python3
"""
generate_arena_map.py

Genera el mapa estático de la arena de pruebas en formato PGM + YAML
para Nav2. El mapa refleja exactamente la geometría del mundo SDF:
  - Arena: 1.22 m × 2.42 m
  - Resolución: 0.02 m/píxel → 61 × 121 píxeles
  - Obstáculo 1: 0.15×0.15 m en (-0.20, +0.10)
  - Obstáculo 2: 0.15×0.15 m en (+0.20, -0.40)
  - Paredes perimetrales

Uso:
  python3 generate_arena_map.py
  # Genera arena_map.pgm y arena_map.yaml en el directorio actual

Después copia los archivos:
  mkdir -p src/omni_dofbot_bringup/maps
  cp arena_map.pgm arena_map.yaml src/omni_dofbot_bringup/maps/
"""

import numpy as np
import os

# ── Parámetros del mapa ────────────────────────────────────────────────────────
RESOLUTION  = 0.02        # metros por píxel
ARENA_W     = 1.22        # metros, eje X
ARENA_H     = 2.42        # metros, eje Y
WALL_T      = 0.03        # grosor de paredes en metros

# Origen del mapa en coordenadas del mundo (esquina inferior izquierda)
# = (-arena_w/2, -arena_h/2)
ORIGIN_X    = -ARENA_W / 2.0   # -0.61
ORIGIN_Y    = -ARENA_H / 2.0   # -1.21

# Tamaño del mapa en píxeles
MAP_W = int(ARENA_W / RESOLUTION)   # 61 px
MAP_H = int(ARENA_H / RESOLUTION)   # 121 px

# Valores PGM: 0=ocupado(negro), 205=desconocido(gris), 254=libre(blanco)
FREE        = 254
OCCUPIED    = 0
UNKNOWN     = 205

# ── Obstáculos (coordenadas mundo en metros) ──────────────────────────────────
OBSTACLES = [
    # (x_center, y_center, width, height)
    (-0.20,  0.10, 0.15, 0.15),   # Obstáculo 1
    ( 0.20, -0.40, 0.15, 0.15),   # Obstáculo 2
]


def world_to_pixel(x_world, y_world):
    """Convierte coordenadas mundo (m) a píxel del mapa (col, row)."""
    col = int((x_world - ORIGIN_X) / RESOLUTION)
    row = int((y_world - ORIGIN_Y) / RESOLUTION)
    return col, row


def draw_rect_world(grid, x_center, y_center, w, h, value, margin=0.0):
    """
    Dibuja un rectángulo en el grid usando coordenadas mundo.
    margin: expansión extra en metros (útil para paredes y obstáculos)
    """
    x_min = x_center - w / 2.0 - margin
    x_max = x_center + w / 2.0 + margin
    y_min = y_center - h / 2.0 - margin
    y_max = y_center + h / 2.0 + margin

    col_min, row_min = world_to_pixel(x_min, y_min)
    col_max, row_max = world_to_pixel(x_max, y_max)

    # Clamp a los límites del mapa
    col_min = max(0, min(MAP_W - 1, col_min))
    col_max = max(0, min(MAP_W - 1, col_max))
    row_min = max(0, min(MAP_H - 1, row_min))
    row_max = max(0, min(MAP_H - 1, row_max))

    grid[row_min:row_max+1, col_min:col_max+1] = value


def generate_map():
    # Inicializar mapa: todo libre
    grid = np.full((MAP_H, MAP_W), FREE, dtype=np.uint8)

    # Paredes perimetrales (grosor WALL_T)
    # Norte
    draw_rect_world(grid, 0, ARENA_H/2 - WALL_T/2, ARENA_W, WALL_T, OCCUPIED)
    # Sur
    draw_rect_world(grid, 0, -ARENA_H/2 + WALL_T/2, ARENA_W, WALL_T, OCCUPIED)
    # Este
    draw_rect_world(grid, ARENA_W/2 - WALL_T/2, 0, WALL_T, ARENA_H, OCCUPIED)
    # Oeste
    draw_rect_world(grid, -ARENA_W/2 + WALL_T/2, 0, WALL_T, ARENA_H, OCCUPIED)

    # Obstáculos con margen de 1 píxel para Nav2
    for (xc, yc, w, h) in OBSTACLES:
        draw_rect_world(grid, xc, yc, w, h, OCCUPIED, margin=RESOLUTION)

    return grid


def save_pgm(grid, path):
    """Guarda el mapa como archivo PGM binario (P5)."""
    h, w = grid.shape
    # Nav2 espera el origen en la esquina inferior izquierda,
    # pero PGM tiene (0,0) en la esquina superior izquierda → invertir Y
    flipped = np.flipud(grid)
    with open(path, 'wb') as f:
        # Header PGM
        f.write(f'P5\n{w} {h}\n255\n'.encode())
        f.write(flipped.tobytes())
    print(f'  Mapa PGM guardado: {path}  ({w}×{h} px)')


def save_yaml(path, pgm_filename):
    """Guarda el archivo YAML de metadatos del mapa."""
    content = f"""# arena_map.yaml
# Mapa estático de la arena de pruebas para Nav2.
# Generado por generate_arena_map.py
#
# Resolución: {RESOLUTION} m/px
# Tamaño: {MAP_W} × {MAP_H} px = {ARENA_W} × {ARENA_H} m
# Origen: esquina inferior izquierda del mapa en coordenadas mundo

image: {pgm_filename}
resolution: {RESOLUTION}
origin: [{ORIGIN_X}, {ORIGIN_Y}, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.25
"""
    with open(path, 'w') as f:
        f.write(content)
    print(f'  YAML guardado: {path}')


def main():
    print('Generando mapa de la arena...')
    print(f'  Tamaño: {MAP_W} × {MAP_H} px ({ARENA_W} × {ARENA_H} m)')
    print(f'  Resolución: {RESOLUTION} m/px')
    print(f'  Origen mundo: ({ORIGIN_X}, {ORIGIN_Y})')

    grid = generate_map()

    pgm_path  = 'arena_map.pgm'
    yaml_path = 'arena_map.yaml'

    save_pgm(grid, pgm_path)
    save_yaml(yaml_path, pgm_path)

    # Estadísticas
    free_cells = np.sum(grid == FREE)
    occ_cells  = np.sum(grid == OCCUPIED)
    total      = grid.size
    print(f'  Celdas libres:   {free_cells} ({100*free_cells/total:.1f}%)')
    print(f'  Celdas ocupadas: {occ_cells}  ({100*occ_cells/total:.1f}%)')
    print()
    print('Copia los archivos al paquete:')
    print('  mkdir -p src/omni_dofbot_bringup/maps')
    print('  cp arena_map.pgm arena_map.yaml src/omni_dofbot_bringup/maps/')


if __name__ == '__main__':
    main()