import os
import sys

# Añadir lerobot src al path
current_dir = os.path.dirname(os.path.abspath(__file__))
lerobot_src = os.path.join(current_dir, "lerobot", "src")
sys.path.insert(0, lerobot_src)

from lerobot.policies.smolvla_layercut.modeling_smolvla_layercut import SmolVLALayercutPolicy
import torch

def test_layercut():
    print("[*] Probando smolvla_layercut con capas [0, 1] desactivadas...")
    policy = SmolVLALayercutPolicy.from_pretrained("lerobot/smolvla_base", ablate_layer_indices=[0, 1])
    
    print(f"Config ablate_layer_indices: {policy.config.ablate_layer_indices}")
    
    expert_layers = policy.model.vlm_with_expert.lm_expert.layers
    models = [policy.model.vlm_with_expert.get_vlm_model().text_model, policy.model.vlm_with_expert.lm_expert]
    vlm_layers, expert_layers_patched = policy.model.vlm_with_expert.get_model_layers(models)
    
    print(f"Total de capas del expert: {len(expert_layers_patched)}")
    print(f"Capa 0: {expert_layers_patched[0]}")
    print(f"Capa 1: {expert_layers_patched[1]}")
    print(f"Capa 2: {'None' if expert_layers_patched[2] is None else 'Not None'}")
    
    assert expert_layers_patched[0] is None, "La capa 0 no ha sido bypasseada"
    assert expert_layers_patched[1] is None, "La capa 1 no ha sido bypasseada"
    assert expert_layers_patched[2] is not None, "La capa 2 no deberia ser bypasseada"
    
    print("[*] Prueba con ablacion superada.")
    
def test_layercut_empty():
    print("[*] Probando smolvla_layercut sin ablación (comportamiento vanilla)...")
    policy = SmolVLALayercutPolicy.from_pretrained("lerobot/smolvla_base")
    
    models = [policy.model.vlm_with_expert.get_vlm_model().text_model, policy.model.vlm_with_expert.lm_expert]
    vlm_layers, expert_layers_patched = policy.model.vlm_with_expert.get_model_layers(models)
    
    assert expert_layers_patched[0] is not None, "La capa 0 ha sido bypasseada"
    
    print("[*] Prueba sin ablacion superada.")

if __name__ == "__main__":
    test_layercut()
    test_layercut_empty()
    print("[*] ¡Todo funciona correctamente!")
