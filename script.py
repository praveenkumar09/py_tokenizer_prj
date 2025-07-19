# This is a sample Python script.
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
model_name = "facebook/blenderbot-400M-distill"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name);
tokenizer = AutoTokenizer.from_pretrained(model_name);

def chat_with_bot():
    # Use a breakpoint in the code line below to debug your script.
        input_text = input("Enter your message: ");
        inputs = tokenizer.encode(input_text, return_tensors="pt");
        outputs = model.generate(inputs, max_new_tokens=150);
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


        print("Chatbot:", response)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    chat_with_bot()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
