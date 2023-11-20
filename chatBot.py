from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch
import warnings

# Filtrer les avertissements liés à Hugging Face et Transformers
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
config = GPT2Config.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, config=config)

# Définir pad_token_id sur eos_token_id
model.config.pad_token_id = tokenizer.eos_token_id

# Supprimer les avertissements liés à la génération du modèle
with torch.no_grad():
    fake_input = torch.tensor([[1]])
    _ = model(fake_input)  # Utilisez une entrée factice pour initialiser les poids

while True:
    user_input = input("Vous: ")
    
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    
    # Ajouter l'attention_mask
    attention_mask = input_ids.ne(model.config.pad_token_id).long()
    
    # Générer la réponse du modèle
    output = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7, attention_mask=attention_mask)
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Afficher l'input de l'utilisateur et la réponse du chatbot
    print(f"ChatBot: {response}\n")
