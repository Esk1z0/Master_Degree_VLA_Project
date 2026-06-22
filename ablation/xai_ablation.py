#!/usr/bin/env python3
import os
import sys
import types
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Añadir el directorio src de lerobot al PYTHONPATH local para asegurar que se importa correctamente
current_dir = os.path.dirname(os.path.abspath(__file__))
lerobot_src = os.path.join(current_dir, "lerobot", "src")
if os.path.exists(lerobot_src):
    sys.path.insert(0, lerobot_src)
    print(f"[*] Añadido al PYTHONPATH: {lerobot_src}")

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.types import FeatureType

def generate_mock_batch(policy, batch_size=1):
    """
    Genera un batch aleatorio dinámicamente basado en la configuración 
    de entradas esperadas por la política SmolVLA.
    """
    device = next(policy.parameters()).device
    batch = {}
    
    print("[*] Detectando características requeridas de la política:")
    
    # 1. Detectar imágenes
    for key, feature in policy.config.input_features.items():
        if feature.type == FeatureType.VISUAL:
            # shape del feature suele ser (C, H, W), ej. (3, 480, 640)
            # El batch espera (B, C, H, W)
            shape = (batch_size, *feature.shape)
            batch[key] = torch.rand(shape, device=device, dtype=torch.float32)
            print(f"  - Imagen: '{key}' con forma {shape}")
            
    # 2. Detectar estado del robot (Joints/Gripper)
    if "observation.state" in policy.config.input_features:
        state_feature = policy.config.input_features["observation.state"]
        shape = (batch_size, *state_feature.shape)
        batch["observation.state"] = torch.rand(shape, device=device, dtype=torch.float32)
        print(f"  - Estado: 'observation.state' con forma {shape}")
        
    # 3. Lenguaje / Instrucciones
    max_length = policy.config.tokenizer_max_length
    batch["observation.language.tokens"] = torch.randint(
        0, 1000, (batch_size, max_length), device=device, dtype=torch.long
    )
    batch["observation.language.attention_mask"] = torch.ones(
        (batch_size, max_length), device=device, dtype=torch.bool
    )
    print(f"  - Lenguaje: Tokens e instruccion (max_length={max_length})")
    
    return batch

def run_ablation_for_layers(policy, batch, ablate_layer_indices):
    """
    Bypassa (ablasiona) sistemáticamente las capas en `ablate_layer_indices`.
    Utiliza el mecanismo integrado de LeRobot donde si una capa en `expert_layers` 
    es None, se salta limpiamente aplicando una conexión identidad sin alterar el tensor.
    """
    vlm_with_expert = policy.model.vlm_with_expert
    original_get_model_layers = vlm_with_expert.get_model_layers
    original_forward_attn_layer = vlm_with_expert.forward_attn_layer
    
    # Definimos la función modificada que inyectará None en las capas seleccionadas
    def ablated_get_model_layers(self, models):
        vlm_layers, expert_layers = original_get_model_layers(models)
        target_layers = [self.lm_expert.layers[i] for i in ablate_layer_indices]
        
        new_expert_layers = []
        for layer in expert_layers:
            if layer in target_layers:
                new_expert_layers.append(None)  # Reemplazado por None (Bypass)
            else:
                new_expert_layers.append(layer)
        return [vlm_layers, new_expert_layers]
        
    def ablated_forward_attn_layer(self, *args, **kwargs):
        try:
            return original_forward_attn_layer(*args, **kwargs)
        except ValueError as e:
            # Capturamos el error de torch.cat con lista vacía cuando saltamos la capa
            if "expected a non-empty list of Tensors" in str(e):
                past_key_values = kwargs.get("past_key_values")
                if past_key_values is None and len(args) >= 10:
                    past_key_values = args[9]
                return [None], past_key_values
            raise e
    
    # Monkeypatch temporal de las funciones
    vlm_with_expert.get_model_layers = types.MethodType(ablated_get_model_layers, vlm_with_expert)
    vlm_with_expert.forward_attn_layer = types.MethodType(ablated_forward_attn_layer, vlm_with_expert)
    
    try:
        # Inferencia con la capa desactivada
        with torch.no_grad():
            ablated_actions = policy.predict_action_chunk(batch)
    finally:
        # Restaurar las funciones originales garantizando que no rompemos el modelo
        vlm_with_expert.get_model_layers = original_get_model_layers
        vlm_with_expert.forward_attn_layer = original_forward_attn_layer
        
    return ablated_actions

def main():
    # 1. Inicializar la política de SmolVLA
    # Puedes cambiar "lerobot/smolvla_base" por tu ruta local si tienes un checkpoint entrenado.
    model_path = "lerobot/smolvla_base"
    print(f"[*] Cargando modelo SmolVLA desde '{model_path}'...")
    
    # Cargar en GPU si está disponible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Ejecutando en dispositivo: {device}")
    
    try:
        policy = SmolVLAPolicy.from_pretrained(model_path)
        policy.to(device)
        policy.eval()
    except Exception as e:
        print(f"[!] Error al cargar el modelo: {e}")
        print("Asegúrate de haber instalado las dependencias de smolvla: pip install -e \".[smolvla]\"")
        return

    # 2. Generar entradas aleatorias (Batch)
    batch = generate_mock_batch(policy, batch_size=1)
    
    # 3. Obtener la predicción base (Baseline)
    print("\n[*] Obteniendo predicciones del modelo base (Baseline)...")
    with torch.no_grad():
        baseline_actions = policy.predict_action_chunk(batch)
    print(f"  - Forma de las acciones obtenidas: {baseline_actions.shape}")

    # 4. Obtener capas del Action Expert
    expert_layers = policy.model.vlm_with_expert.lm_expert.layers
    num_layers = len(expert_layers)
    print(f"[*] Action Expert detectado con {num_layers} capas.")
    
    # Directorio de resultados
    results_dir = os.path.join(current_dir, "results", "layer_basic")
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = []

    # 5. Ejecutar la ablación multicapa (ventana deslizante)
    print("\n[*] Iniciando análisis multicapa de XAI (Ventanas Deslizantes)...")
    for window_size in range(1, num_layers):
        print(f"[*] Evaluando ventana de desactivación: {window_size} capa(s)")
        
        for start_idx in range(num_layers - window_size + 1):
            indices_to_ablate = list(range(start_idx, start_idx + window_size))
            print(f"  - Desactivando capas {indices_to_ablate}...", end="\r")
            
            ablated_actions = run_ablation_for_layers(policy, batch, indices_to_ablate)
            
            # Calcular métricas de cambio en las acciones
            mse = torch.mean((baseline_actions - ablated_actions) ** 2).item()
            mae = torch.mean(torch.abs(baseline_actions - ablated_actions)).item()
            
            all_results.append({
                "window_size": window_size,
                "start_idx": start_idx,
                "ablated_layers": indices_to_ablate,
                "mse": mse,
                "mae": mae
            })
            
        print(f"  - Análisis ventana {window_size} completado.               ")
        
    print("\n[*] Análisis completado con éxito.")
    
    # Guardar resultados en JSON
    json_path = os.path.join(results_dir, "ablation_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=4)
        
    # Generar gráficas
    print(f"[*] Generando gráficas en {results_dir} ...")
    
    # 1. Heatmap para MSE
    heatmap_data_mse = np.zeros((num_layers, num_layers))
    heatmap_data_mse[:] = np.nan
    for res in all_results:
        heatmap_data_mse[res["window_size"], res["start_idx"]] = res["mse"]
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data_mse[1:, :], annot=False, cmap="viridis", 
                yticklabels=range(1, num_layers), xticklabels=range(num_layers))
    plt.title("Error Cuadrático Medio (MSE) por Ventana de Ablación")
    plt.xlabel("Índice de Capa Inicial (start_idx)")
    plt.ylabel("Tamaño de la Ventana (Número de Capas Desactivadas)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "mse_heatmap.png"))
    plt.close()

    # 2. Gráfico de líneas para window_size = 1, 2 y 3
    plt.figure(figsize=(10, 6))
    for ws in [1, 2, 3]:
        ws_data = [res for res in all_results if res["window_size"] == ws]
        if ws_data:
            x = [res["start_idx"] for res in ws_data]
            y = [res["mse"] for res in ws_data]
            plt.plot(x, y, marker='o', label=f'Ventana = {ws}')
    plt.title("Evolución del MSE para diferentes tamaños de ventana")
    plt.xlabel("Índice de Capa Inicial")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "mse_lines.png"))
    plt.close()
    
    # 3. Mostrar resumen de los resultados de ventana 1
    single_layer_results = [res for res in all_results if res["window_size"] == 1]
    if single_layer_results:
        mse_values = [res["mse"] for res in single_layer_results]
        idx_most_critical = np.argmax(mse_values)
        idx_least_critical = np.argmin(mse_values)
        print("\n[🎯 DIAGNÓSTICO XAI (Capa Individual)]")
        print(f"  - Capa MÁS crítica: Capa {idx_most_critical} (MSE: {mse_values[idx_most_critical]:.6e})")
        print(f"  - Capa MENOS crítica: Capa {idx_least_critical} (MSE: {mse_values[idx_least_critical]:.6e})")

    print("[*] ¡Análisis completado y guardado!")

if __name__ == "__main__":
    main()
