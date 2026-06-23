# Plan de Estudio de Ablación de Capas - SmolVLA Action Expert

Este documento detalla todas las combinaciones de capas que deben desactivarse para realizar el estudio de degradación del modelo **SmolVLA**, específicamente en su módulo **Action Expert** (16 capas).

## Nomenclatura Propuesta
- **Modelo Base:** `smolvla_base`
- **Modelos Ablacionados:** `smolvla_ablation_w{tamaño}_s{inicio}`
  - `w` (Window): Indica el tamaño del bloque de capas desactivadas.
  - `s` (Start): Indica el índice de la primera capa del bloque.

## Parámetros del Estudio
- **Total de Capas:** 16 (índices del 0 al 15).
- **Estrategia:** Ventanas deslizantes (sliding window).
- **Incremento de Tamaño:** Desde 1 capa hasta 15 capas.
- **Stride (Paso):** 1.
- **Total de Ejecuciones / Modelos:** 120.

---

### Pasada 1: Bloques de tamaño 1
*(Total de ejecuciones en esta pasada: 16)*

| Nombre del Modelo / Ejecución | Capas a Desactivar (Índices) |
|---|---|
| `smolvla_ablation_w1_s0` | `[0]` |
| `smolvla_ablation_w1_s1` | `[1]` |
| `smolvla_ablation_w1_s2` | `[2]` |
| `smolvla_ablation_w1_s3` | `[3]` |
| `smolvla_ablation_w1_s4` | `[4]` |
| `smolvla_ablation_w1_s5` | `[5]` |
| `smolvla_ablation_w1_s6` | `[6]` |
| `smolvla_ablation_w1_s7` | `[7]` |
| `smolvla_ablation_w1_s8` | `[8]` |
| `smolvla_ablation_w1_s9` | `[9]` |
| `smolvla_ablation_w1_s10` | `[10]` |
| `smolvla_ablation_w1_s11` | `[11]` |
| `smolvla_ablation_w1_s12` | `[12]` |
| `smolvla_ablation_w1_s13` | `[13]` |
| `smolvla_ablation_w1_s14` | `[14]` |
| `smolvla_ablation_w1_s15` | `[15]` |

### Pasada 2: Bloques de tamaño 2
*(Total de ejecuciones en esta pasada: 15)*

| Nombre del Modelo / Ejecución | Capas a Desactivar (Índices) |
|---|---|
| `smolvla_ablation_w2_s0` | `[0, 1]` |
| `smolvla_ablation_w2_s1` | `[1, 2]` |
| `smolvla_ablation_w2_s2` | `[2, 3]` |
| `smolvla_ablation_w2_s3` | `[3, 4]` |
| `smolvla_ablation_w2_s4` | `[4, 5]` |
| `smolvla_ablation_w2_s5` | `[5, 6]` |
| `smolvla_ablation_w2_s6` | `[6, 7]` |
| `smolvla_ablation_w2_s7` | `[7, 8]` |
| `smolvla_ablation_w2_s8` | `[8, 9]` |
| `smolvla_ablation_w2_s9` | `[9, 10]` |
| `smolvla_ablation_w2_s10` | `[10, 11]` |
| `smolvla_ablation_w2_s11` | `[11, 12]` |
| `smolvla_ablation_w2_s12` | `[12, 13]` |
| `smolvla_ablation_w2_s13` | `[13, 14]` |
| `smolvla_ablation_w2_s14` | `[14, 15]` |

### Pasada 3: Bloques de tamaño 3
*(Total de ejecuciones en esta pasada: 14)*

| Nombre del Modelo / Ejecución | Capas a Desactivar (Índices) |
|---|---|
| `smolvla_ablation_w3_s0` | `[0, 1, 2]` |
| `smolvla_ablation_w3_s1` | `[1, 2, 3]` |
| `smolvla_ablation_w3_s2` | `[2, 3, 4]` |
| `smolvla_ablation_w3_s3` | `[3, 4, 5]` |
| `smolvla_ablation_w3_s4` | `[4, 5, 6]` |
| `smolvla_ablation_w3_s5` | `[5, 6, 7]` |
| `smolvla_ablation_w3_s6` | `[6, 7, 8]` |
| `smolvla_ablation_w3_s7` | `[7, 8, 9]` |
| `smolvla_ablation_w3_s8` | `[8, 9, 10]` |
| `smolvla_ablation_w3_s9` | `[9, 10, 11]` |
| `smolvla_ablation_w3_s10` | `[10, 11, 12]` |
| `smolvla_ablation_w3_s11` | `[11, 12, 13]` |
| `smolvla_ablation_w3_s12` | `[12, 13, 14]` |
| `smolvla_ablation_w3_s13` | `[13, 14, 15]` |

### Pasada 4: Bloques de tamaño 4
*(Total de ejecuciones en esta pasada: 13)*

| Nombre del Modelo / Ejecución | Capas a Desactivar (Índices) |
|---|---|
| `smolvla_ablation_w4_s0` | `[0, 1, 2, 3]` |
| `smolvla_ablation_w4_s1` | `[1, 2, 3, 4]` |
| `smolvla_ablation_w4_s2` | `[2, 3, 4, 5]` |
| `smolvla_ablation_w4_s3` | `[3, 4, 5, 6]` |
| `smolvla_ablation_w4_s4` | `[4, 5, 6, 7]` |
| `smolvla_ablation_w4_s5` | `[5, 6, 7, 8]` |
| `smolvla_ablation_w4_s6` | `[6, 7, 8, 9]` |
| `smolvla_ablation_w4_s7` | `[7, 8, 9, 10]` |
| `smolvla_ablation_w4_s8` | `[8, 9, 10, 11]` |
| `smolvla_ablation_w4_s9` | `[9, 10, 11, 12]` |
| `smolvla_ablation_w4_s10` | `[10, 11, 12, 13]` |
| `smolvla_ablation_w4_s11` | `[11, 12, 13, 14]` |
| `smolvla_ablation_w4_s12` | `[12, 13, 14, 15]` |

### Pasada 5: Bloques de tamaño 5
*(Total de ejecuciones en esta pasada: 12)*

| Nombre del Modelo / Ejecución | Capas a Desactivar (Índices) |
|---|---|
| `smolvla_ablation_w5_s0` | `[0, 1, 2, 3, 4]` |
| `smolvla_ablation_w5_s1` | `[1, 2, 3, 4, 5]` |
| `smolvla_ablation_w5_s2` | `[2, 3, 4, 5, 6]` |
| `smolvla_ablation_w5_s3` | `[3, 4, 5, 6, 7]` |
| `smolvla_ablation_w5_s4` | `[4, 5, 6, 7, 8]` |
| `smolvla_ablation_w5_s5` | `[5, 6, 7, 8, 9]` |
| `smolvla_ablation_w5_s6` | `[6, 7, 8, 9, 10]` |
| `smolvla_ablation_w5_s7` | `[7, 8, 9, 10, 11]` |
| `smolvla_ablation_w5_s8` | `[8, 9, 10, 11, 12]` |
| `smolvla_ablation_w5_s9` | `[9, 10, 11, 12, 13]` |
| `smolvla_ablation_w5_s10` | `[10, 11, 12, 13, 14]` |
| `smolvla_ablation_w5_s11` | `[11, 12, 13, 14, 15]` |

### Pasada 6: Bloques de tamaño 6
*(Total de ejecuciones en esta pasada: 11)*

| Nombre del Modelo / Ejecución | Capas a Desactivar (Índices) |
|---|---|
| `smolvla_ablation_w6_s0` | `[0, 1, 2, 3, 4, 5]` |
| `smolvla_ablation_w6_s1` | `[1, 2, 3, 4, 5, 6]` |
| `smolvla_ablation_w6_s2` | `[2, 3, 4, 5, 6, 7]` |
| `smolvla_ablation_w6_s3` | `[3, 4, 5, 6, 7, 8]` |
| `smolvla_ablation_w6_s4` | `[4, 5, 6, 7, 8, 9]` |
| `smolvla_ablation_w6_s5` | `[5, 6, 7, 8, 9, 10]` |
| `smolvla_ablation_w6_s6` | `[6, 7, 8, 9, 10, 11]` |
| `smolvla_ablation_w6_s7` | `[7, 8, 9, 10, 11, 12]` |
| `smolvla_ablation_w6_s8` | `[8, 9, 10, 11, 12, 13]` |
| `smolvla_ablation_w6_s9` | `[9, 10, 11, 12, 13, 14]` |
| `smolvla_ablation_w6_s10` | `[10, 11, 12, 13, 14, 15]` |

### Pasada 7: Bloques de tamaño 7
*(Total de ejecuciones en esta pasada: 10)*

| Nombre del Modelo / Ejecución | Capas a Desactivar (Índices) |
|---|---|
| `smolvla_ablation_w7_s0` | `[0, 1, 2, 3, 4, 5, 6]` |
| `smolvla_ablation_w7_s1` | `[1, 2, 3, 4, 5, 6, 7]` |
| `smolvla_ablation_w7_s2` | `[2, 3, 4, 5, 6, 7, 8]` |
| `smolvla_ablation_w7_s3` | `[3, 4, 5, 6, 7, 8, 9]` |
| `smolvla_ablation_w7_s4` | `[4, 5, 6, 7, 8, 9, 10]` |
| `smolvla_ablation_w7_s5` | `[5, 6, 7, 8, 9, 10, 11]` |
| `smolvla_ablation_w7_s6` | `[6, 7, 8, 9, 10, 11, 12]` |
| `smolvla_ablation_w7_s7` | `[7, 8, 9, 10, 11, 12, 13]` |
| `smolvla_ablation_w7_s8` | `[8, 9, 10, 11, 12, 13, 14]` |
| `smolvla_ablation_w7_s9` | `[9, 10, 11, 12, 13, 14, 15]` |

### Pasada 8: Bloques de tamaño 8
*(Total de ejecuciones en esta pasada: 9)*

| Nombre del Modelo / Ejecución | Capas a Desactivar (Índices) |
|---|---|
| `smolvla_ablation_w8_s0` | `[0, 1, 2, 3, 4, 5, 6, 7]` |
| `smolvla_ablation_w8_s1` | `[1, 2, 3, 4, 5, 6, 7, 8]` |
| `smolvla_ablation_w8_s2` | `[2, 3, 4, 5, 6, 7, 8, 9]` |
| `smolvla_ablation_w8_s3` | `[3, 4, 5, 6, 7, 8, 9, 10]` |
| `smolvla_ablation_w8_s4` | `[4, 5, 6, 7, 8, 9, 10, 11]` |
| `smolvla_ablation_w8_s5` | `[5, 6, 7, 8, 9, 10, 11, 12]` |
| `smolvla_ablation_w8_s6` | `[6, 7, 8, 9, 10, 11, 12, 13]` |
| `smolvla_ablation_w8_s7` | `[7, 8, 9, 10, 11, 12, 13, 14]` |
| `smolvla_ablation_w8_s8` | `[8, 9, 10, 11, 12, 13, 14, 15]` |

### Pasada 9: Bloques de tamaño 9
*(Total de ejecuciones en esta pasada: 8)*

| Nombre del Modelo / Ejecución | Capas a Desactivar (Índices) |
|---|---|
| `smolvla_ablation_w9_s0` | `[0, 1, 2, 3, 4, 5, 6, 7, 8]` |
| `smolvla_ablation_w9_s1` | `[1, 2, 3, 4, 5, 6, 7, 8, 9]` |
| `smolvla_ablation_w9_s2` | `[2, 3, 4, 5, 6, 7, 8, 9, 10]` |
| `smolvla_ablation_w9_s3` | `[3, 4, 5, 6, 7, 8, 9, 10, 11]` |
| `smolvla_ablation_w9_s4` | `[4, 5, 6, 7, 8, 9, 10, 11, 12]` |
| `smolvla_ablation_w9_s5` | `[5, 6, 7, 8, 9, 10, 11, 12, 13]` |
| `smolvla_ablation_w9_s6` | `[6, 7, 8, 9, 10, 11, 12, 13, 14]` |
| `smolvla_ablation_w9_s7` | `[7, 8, 9, 10, 11, 12, 13, 14, 15]` |

### Pasada 10: Bloques de tamaño 10
*(Total de ejecuciones en esta pasada: 7)*

| Nombre del Modelo / Ejecución | Capas a Desactivar (Índices) |
|---|---|
| `smolvla_ablation_w10_s0` | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` |
| `smolvla_ablation_w10_s1` | `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]` |
| `smolvla_ablation_w10_s2` | `[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]` |
| `smolvla_ablation_w10_s3` | `[3, 4, 5, 6, 7, 8, 9, 10, 11, 12]` |
| `smolvla_ablation_w10_s4` | `[4, 5, 6, 7, 8, 9, 10, 11, 12, 13]` |
| `smolvla_ablation_w10_s5` | `[5, 6, 7, 8, 9, 10, 11, 12, 13, 14]` |
| `smolvla_ablation_w10_s6` | `[6, 7, 8, 9, 10, 11, 12, 13, 14, 15]` |

### Pasada 11: Bloques de tamaño 11
*(Total de ejecuciones en esta pasada: 6)*

| Nombre del Modelo / Ejecución | Capas a Desactivar (Índices) |
|---|---|
| `smolvla_ablation_w11_s0` | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]` |
| `smolvla_ablation_w11_s1` | `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]` |
| `smolvla_ablation_w11_s2` | `[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]` |
| `smolvla_ablation_w11_s3` | `[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]` |
| `smolvla_ablation_w11_s4` | `[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]` |
| `smolvla_ablation_w11_s5` | `[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]` |

### Pasada 12: Bloques de tamaño 12
*(Total de ejecuciones en esta pasada: 5)*

| Nombre del Modelo / Ejecución | Capas a Desactivar (Índices) |
|---|---|
| `smolvla_ablation_w12_s0` | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]` |
| `smolvla_ablation_w12_s1` | `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]` |
| `smolvla_ablation_w12_s2` | `[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]` |
| `smolvla_ablation_w12_s3` | `[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]` |
| `smolvla_ablation_w12_s4` | `[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]` |

### Pasada 13: Bloques de tamaño 13
*(Total de ejecuciones en esta pasada: 4)*

| Nombre del Modelo / Ejecución | Capas a Desactivar (Índices) |
|---|---|
| `smolvla_ablation_w13_s0` | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]` |
| `smolvla_ablation_w13_s1` | `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]` |
| `smolvla_ablation_w13_s2` | `[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]` |
| `smolvla_ablation_w13_s3` | `[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]` |

### Pasada 14: Bloques de tamaño 14
*(Total de ejecuciones en esta pasada: 3)*

| Nombre del Modelo / Ejecución | Capas a Desactivar (Índices) |
|---|---|
| `smolvla_ablation_w14_s0` | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]` |
| `smolvla_ablation_w14_s1` | `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]` |
| `smolvla_ablation_w14_s2` | `[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]` |

### Pasada 15: Bloques de tamaño 15
*(Total de ejecuciones en esta pasada: 2)*

| Nombre del Modelo / Ejecución | Capas a Desactivar (Índices) |
|---|---|
| `smolvla_ablation_w15_s0` | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]` |
| `smolvla_ablation_w15_s1` | `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]` |
