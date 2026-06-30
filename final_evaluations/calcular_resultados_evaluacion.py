import csv
import sys
import os
import math

def safe_float(val):
    """Convierte de forma segura un valor a flotante. Retorna None si está vacío o es inválido."""
    if val is None or str(val).strip() == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None

def fmt(val, decimals=4):
    """Formatea valores numéricos limitando decimales y devuelve cadena vacía si es None."""
    if val is None:
        return ''
    return round(val, decimals)

def procesar_archivo(file_path):
    """Procesa un archivo de evaluación y calcula las métricas derivadas."""
    base_name = os.path.basename(file_path)
    model_name, _ = os.path.splitext(base_name)
    
    processed_rows = []
    file_results = []
    
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Validar columnas obligatorias
        required_cols = [
            'test_id', 'escenario', 'setup_validado', 'objeto_visible_inicio', 'oclusion_verificada',
            'estrella_negra_agarre', 'estrella_negra_destino', 'estrella_naranja_agarre', 'estrella_naranja_destino',
            'cubo_negro_agarre', 'cubo_negro_destino', 'cubo_naranja_agarre', 'cubo_naranja_destino',
            'finalizacion_limpia', 'recuerda_objeto_ocluido', 'completa_objeto_ocluido',
            'selecciona_real_antes_foto', 'evita_intento_agarre_foto', 'aproximaciones_foto', 'intentos_agarre_foto',
            'tiempo_total_s', 'timeout', 'intervencion_emergencia'
        ]
        missing_cols = [c for c in required_cols if c not in reader.fieldnames]
        if missing_cols:
            print(f"Error: Al archivo {file_path} le faltan las columnas obligatorias: {missing_cols}")
            return None, None
            
        for row_idx, row in enumerate(reader, start=2):
            escenario = row.get('escenario', '')
            
            setup_validado = safe_float(row.get('setup_validado'))
            objeto_visible_inicio = safe_float(row.get('objeto_visible_inicio'))
            oclusion_verificada = safe_float(row.get('oclusion_verificada'))
            
            # Validez del registro
            registro_valido = 0
            if escenario in ['autooclusion', 'combinado']:
                if setup_validado == 1 and objeto_visible_inicio == 1 and oclusion_verificada == 1:
                    registro_valido = 1
            else:
                if setup_validado == 1:
                    registro_valido = 1
            
            row['registro_valido'] = registro_valido

            if registro_valido == 0:
                print(f"Advertencia: El registro {row.get('test_id')} (fila {row_idx}) en {file_path} no es válido.")

            # Destreza
            destreza_cols = [
                'estrella_negra_agarre', 'estrella_negra_destino',
                'estrella_naranja_agarre', 'estrella_naranja_destino',
                'cubo_negro_agarre', 'cubo_negro_destino',
                'cubo_naranja_agarre', 'cubo_naranja_destino',
                'finalizacion_limpia'
            ]
            puntuacion_destreza = sum([1 if safe_float(row.get(c)) == 1 else 0 for c in destreza_cols])
            max_destreza = 9
            
            # Memoria
            if escenario in ['autooclusion', 'combinado']:
                puntuacion_memoria = sum([1 if safe_float(row.get(c)) == 1 else 0 for c in ['recuerda_objeto_ocluido', 'completa_objeto_ocluido']])
                max_memoria = 2
            else:
                puntuacion_memoria = 0
                max_memoria = 0
            
            # 3D (Fotografía)
            if escenario in ['fotografia', 'combinado']:
                puntuacion_3d = sum([1 if safe_float(row.get(c)) == 1 else 0 for c in ['selecciona_real_antes_foto', 'evita_intento_agarre_foto']])
                max_3d = 2
            else:
                puntuacion_3d = 0
                max_3d = 0
            
            # Totales
            puntuacion_total = puntuacion_destreza + puntuacion_memoria + puntuacion_3d
            puntuacion_max = max_destreza + max_memoria + max_3d
            puntuacion_normalizada = puntuacion_total / puntuacion_max if puntuacion_max > 0 else 0
            
            timeout = 1 if safe_float(row.get('timeout')) == 1 else 0
            intervencion_emergencia = 1 if safe_float(row.get('intervencion_emergencia')) == 1 else 0
            
            exito_completo = 1 if (registro_valido == 1 and puntuacion_total == puntuacion_max and timeout == 0 and intervencion_emergencia == 0) else 0

            # Guardar en row y procesar
            row['puntuacion_destreza'] = puntuacion_destreza
            row['max_destreza'] = max_destreza
            row['puntuacion_memoria'] = puntuacion_memoria
            row['max_memoria'] = max_memoria
            row['puntuacion_3d'] = puntuacion_3d
            row['max_3d'] = max_3d
            row['puntuacion_total'] = puntuacion_total
            row['puntuacion_max'] = puntuacion_max
            row['puntuacion_normalizada'] = puntuacion_normalizada
            row['exito_completo'] = exito_completo
            
            processed_rows.append(row)
            
            # Para la agregación
            metrics = {
                'modelo': model_name,
                'escenario': escenario,
                'registro_valido': registro_valido,
                'puntuacion_total': puntuacion_total,
                'puntuacion_normalizada': puntuacion_normalizada,
                'exito_completo': exito_completo,
                'timeout': timeout,
                'intervencion_emergencia': intervencion_emergencia,
                'tiempo_total_s': safe_float(row.get('tiempo_total_s')),
                'aproximaciones_foto': safe_float(row.get('aproximaciones_foto')),
                'intentos_agarre_foto': safe_float(row.get('intentos_agarre_foto')),
                'evita_intento_agarre_foto': safe_float(row.get('evita_intento_agarre_foto')),
                'recuerda_objeto_ocluido': safe_float(row.get('recuerda_objeto_ocluido')),
                'completa_objeto_ocluido': safe_float(row.get('completa_objeto_ocluido'))
            }
            file_results.append(metrics)
            
    # Escribir archivo de resultados individuales
    if processed_rows:
        output_file = f"{model_name}_calculado.csv"
        # Mantener el directorio original
        output_path = os.path.join(os.path.dirname(file_path), output_file)
        new_fieldnames = reader.fieldnames + [
            'puntuacion_destreza', 'max_destreza', 'puntuacion_memoria', 'max_memoria',
            'puntuacion_3d', 'max_3d', 'puntuacion_total', 'puntuacion_max',
            'puntuacion_normalizada', 'registro_valido', 'exito_completo'
        ]
        with open(output_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=new_fieldnames)
            writer.writeheader()
            writer.writerows(processed_rows)
        print(f"Guardado archivo de resultados: {output_path}")
        
    return file_results, processed_rows


def generar_resumen(all_results, output_dir):
    """Calcula estadísticas y genera el resumen global cuando hay varios modelos."""
    summary_rows = []
    models = sorted(list(set([r['modelo'] for r in all_results])))
    escenarios = ['destreza_general', 'autooclusion', 'fotografia', 'combinado']
    
    for model in models:
        model_results = [r for r in all_results if r['modelo'] == model]
        
        for esc in escenarios + ['global']:
            if esc == 'global':
                rows = model_results
            else:
                rows = [r for r in model_results if r['escenario'] == esc]
            
            if not rows:
                continue
            
            n_total = len(rows)
            valid_rows = [r for r in rows if r['registro_valido'] == 1]
            n_validos = len(valid_rows)
            
            if n_validos > 0:
                puntuaciones_tot = [r['puntuacion_total'] for r in valid_rows]
                puntuacion_media = sum(puntuaciones_tot) / n_validos
                
                puntuaciones_norm = [r['puntuacion_normalizada'] for r in valid_rows]
                puntuacion_normalizada_media = sum(puntuaciones_norm) / n_validos
                
                var_sum = sum([(p - puntuacion_normalizada_media) ** 2 for p in puntuaciones_norm])
                desviacion_estandar_normalizada = math.sqrt(var_sum / n_validos) if n_validos > 1 else 0.0
                
                exitos_completos = sum([r['exito_completo'] for r in valid_rows])
                tasa_exito = exitos_completos / n_validos
                
                timeouts = sum([r['timeout'] for r in rows])
                intervenciones_emergencia = sum([r['intervencion_emergencia'] for r in rows])
                
                tiempos = [r['tiempo_total_s'] for r in valid_rows if r['tiempo_total_s'] is not None]
                tiempo_medio_s = sum(tiempos) / len(tiempos) if tiempos else None
                
                # Métricas 3D
                if esc in ['fotografia', 'combinado', 'global']:
                    aprox = [r['aproximaciones_foto'] for r in valid_rows if r['aproximaciones_foto'] is not None]
                    aproximaciones_foto = sum(aprox) / len(aprox) if aprox else None
                    
                    intentos = [r['intentos_agarre_foto'] for r in valid_rows if r['intentos_agarre_foto'] is not None]
                    intentos_agarre_foto = sum(intentos) / len(intentos) if intentos else None
                    
                    evitas = [r['evita_intento_agarre_foto'] for r in valid_rows if r['evita_intento_agarre_foto'] is not None]
                    tasa_rechazo_foto = sum(evitas) / len(evitas) if evitas else None
                else:
                    aproximaciones_foto = intentos_agarre_foto = tasa_rechazo_foto = None
                    
                # Métricas memoria
                if esc in ['autooclusion', 'combinado', 'global']:
                    recuerdos = [r['recuerda_objeto_ocluido'] for r in valid_rows if r['recuerda_objeto_ocluido'] is not None]
                    tasa_recuerdo_ocluido = sum(recuerdos) / len(recuerdos) if recuerdos else None
                    
                    completados = [r['completa_objeto_ocluido'] for r in valid_rows if r['completa_objeto_ocluido'] is not None]
                    tasa_completado_ocluido = sum(completados) / len(completados) if completados else None
                else:
                    tasa_recuerdo_ocluido = tasa_completado_ocluido = None
            else:
                puntuacion_media = 0
                puntuacion_normalizada_media = 0
                desviacion_estandar_normalizada = 0
                exitos_completos = 0
                tasa_exito = 0
                timeouts = sum([r['timeout'] for r in rows])
                intervenciones_emergencia = sum([r['intervencion_emergencia'] for r in rows])
                tiempo_medio_s = None
                aproximaciones_foto = intentos_agarre_foto = tasa_rechazo_foto = None
                tasa_recuerdo_ocluido = tasa_completado_ocluido = None

            summary_rows.append({
                'modelo': model,
                'escenario': esc,
                'n_total': n_total,
                'n_validos': n_validos,
                'puntuacion_media': fmt(puntuacion_media, 2),
                'puntuacion_normalizada_media': fmt(puntuacion_normalizada_media, 4),
                'desviacion_estandar_normalizada': fmt(desviacion_estandar_normalizada, 4),
                'exitos_completos': exitos_completos,
                'tasa_exito': fmt(tasa_exito, 4),
                'timeouts': timeouts,
                'intervenciones_emergencia': intervenciones_emergencia,
                'aproximaciones_foto': fmt(aproximaciones_foto, 4),
                'intentos_agarre_foto': fmt(intentos_agarre_foto, 4),
                'tasa_rechazo_foto': fmt(tasa_rechazo_foto, 4),
                'tasa_recuerdo_ocluido': fmt(tasa_recuerdo_ocluido, 4),
                'tasa_completado_ocluido': fmt(tasa_completado_ocluido, 4),
                'tiempo_medio_s': fmt(tiempo_medio_s, 2)
            })

    if summary_rows:
        resumen_file = os.path.join(output_dir, "resumen_global_evaluacion.csv")
        with open(resumen_file, mode='w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'modelo', 'escenario', 'n_total', 'n_validos', 'puntuacion_media',
                'puntuacion_normalizada_media', 'desviacion_estandar_normalizada',
                'exitos_completos', 'tasa_exito', 'timeouts', 'intervenciones_emergencia',
                'aproximaciones_foto', 'intentos_agarre_foto', 'tasa_rechazo_foto',
                'tasa_recuerdo_ocluido', 'tasa_completado_ocluido', 'tiempo_medio_s'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Guardado resumen global: {resumen_file}")


def main():
    args = sys.argv[1:]
    if not args:
        print("Uso: python calcular_resultados_evaluacion.py evaluacion_M0.csv [evaluacion_M1.csv ...]")
        sys.exit(1)

    all_results = []
    output_dir = os.path.dirname(os.path.abspath(args[0]))
    
    for file_path in args:
        if not os.path.exists(file_path):
            print(f"Error: El archivo {file_path} no existe.")
            continue
            
        print(f"\nProcesando: {file_path}")
        results, _ = procesar_archivo(file_path)
        if results:
            all_results.extend(results)
            
    if all_results and len(args) > 1:
        generar_resumen(all_results, output_dir)
    print("\nProcesamiento finalizado.")

if __name__ == '__main__':
    main()
