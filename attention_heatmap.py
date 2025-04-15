import seaborn as sns
import matplotlib.pyplot as plt

def pad_sequence(tokens, max_len=50):
    return tokens[:max_len] + [0] * (max_len - len(tokens))

def get_tokens_from_ids(token_ids, sp_model):
    return [sp_model.id_to_piece(token_id) for token_id in token_ids]

def plot_attention(attn_matrix, input_token_ids, output_token_ids, sp_model, pred_sentence):
    input_tokens = get_tokens_from_ids(input_token_ids, sp_model)
    output_token_ids = [token for token in output_token_ids if token != 0]
    output_tokens = get_tokens_from_ids(output_token_ids, sp_model)
    num_columns = len(input_tokens)
    num_rows = len(output_tokens)
    attn_matrix = attn_matrix[:num_rows, :num_columns] 
    plt.figure(figsize=(20, 12))
    sns.heatmap(attn_matrix, cmap="Blues", xticklabels=input_tokens, yticklabels=output_tokens)
    plt.xlabel(f"{input_sentence}")
    plt.ylabel("Output Sequence")
    plt.title(f"{pred_sentence}")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()

def translate_sentence(sentence, model, sp_model, device, max_length=50):
    model.eval()  
    tokens = sp_model.encode(sentence.strip(), out_type=int)  # covert src sentence to token ID
    src_tensor = torch.LongTensor(pad_sequence(tokens)).unsqueeze(0).to(device)

    with torch.no_grad():
        output, attn_weights = model(src_tensor)  
        output_tokens = output.argmax(dim=-1)
        attn_matrix = attn_weights.squeeze(0).squeeze(1).cpu().numpy()  
        input_token_ids = tokens
        output_token_ids = output_tokens.squeeze(0).cpu().numpy().tolist()
        pred_tokens = [t for t in output_tokens[0].cpu().numpy().tolist() if t != 0]
        pred_sentence = sp_model.decode(pred_tokens)
        print(f"Prediction: {pred_sentence}") #print predicted sentence
        # plot
        plot_attention(attn_matrix, input_token_ids, output_token_ids, sp_model, pred_sentence)



# the selected sentence in training
input_sentence = "Ce nouveau raid de l'aviation de Tsahal en territoire syrien (le sixième depuis le début de l'année, selon le quotidien israélien Haaretz) n'a été confirmé ni par Isral ni par la Syrie."
translate_sentence(input_sentence, model, sp_model, device)