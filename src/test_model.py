"""
Quick test to verify the RenikudModel works
"""

import torch
from transformers import AutoTokenizer, AutoConfig

def test_model_initialization():
    """Test that the model can be initialized"""
    print("🧪 Testing model initialization...")
    
    from src.model.renikud_model import RenikudModel
    
    # Create a config
    config = AutoConfig.from_pretrained("dicta-il/dictabert-large-char-menaked", trust_remote_code=True)
    
    # Initialize model
    model = RenikudModel(config)
    print(f"✅ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test freeze
    model.freeze_base_model()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ After freezing base: {trainable:,} trainable parameters")
    
    return model

def test_forward_pass():
    """Test forward pass"""
    print("\n🧪 Testing forward pass...")
    
    from src.model.renikud_model import RenikudModel
    
    config = AutoConfig.from_pretrained("dicta-il/dictabert-large-char-menaked", trust_remote_code=True)
    model = RenikudModel(config)
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 10
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    
    print(f"✅ Vowel logits shape: {output.vowel_logits.shape}")  # (2, 10, 7)
    print(f"✅ Dagesh logits shape: {output.dagesh_logits.shape}")  # (2, 10, 2)
    print(f"✅ Sin logits shape: {output.sin_logits.shape}")  # (2, 10, 2)
    
    assert output.vowel_logits.shape == (batch_size, seq_len, 7)
    assert output.dagesh_logits.shape == (batch_size, seq_len, 2)
    assert output.sin_logits.shape == (batch_size, seq_len, 2)
    
    return True

def test_data_loading():
    """Test data loading"""
    print("\n🧪 Testing data loading...")
    
    from src.data import TrainData
    
    # Create dummy data
    unvocalized = ["שלום", "עולם"]
    vocalized = ["שָׁלוֹם", "עוֹלָם"]
    
    dataset = TrainData(unvocalized, vocalized)
    
    # Get first item
    text, vocalized_text, vowel_labels, dagesh_labels, sin_labels = dataset[0]
    
    print(f"✅ Text: {text}")
    print(f"✅ Vowel labels shape: {vowel_labels.shape}")
    print(f"✅ Dagesh labels shape: {dagesh_labels.shape}")
    print(f"✅ Sin labels shape: {sin_labels.shape}")
    
    return True

def test_constants():
    """Test constants are properly defined"""
    print("\n🧪 Testing constants...")
    
    from src.constants import VOWEL_CLASSES, CAN_HAVE_DAGESH, CAN_HAVE_SIN_DOT
    
    print(f"✅ VOWEL_CLASSES: {len(VOWEL_CLASSES)} classes")
    print(f"   {VOWEL_CLASSES}")
    print(f"✅ CAN_HAVE_DAGESH: {CAN_HAVE_DAGESH}")
    print(f"✅ CAN_HAVE_SIN_DOT: {CAN_HAVE_SIN_DOT}")
    
    assert len(VOWEL_CLASSES) == 7  # empty + 6 vowels
    assert len(CAN_HAVE_DAGESH) == 6  # בכךפףו
    assert len(CAN_HAVE_SIN_DOT) == 1  # ש
    
    return True

if __name__ == "__main__":
    print("🚀 Running Renikud model tests\n")
    
    try:
        test_constants()
        test_model_initialization()
        test_forward_pass()
        test_data_loading()
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

