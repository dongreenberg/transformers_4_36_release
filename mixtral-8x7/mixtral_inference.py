import runhouse as rh
from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM
import torch


class HFChatModel(rh.Module):
    def __init__(self, model_id="meta-llama/Llama-2-13b-chat-hf", **model_kwargs):
        super().__init__()
        self.model_id, self.model_kwargs = model_id, model_kwargs
        self.tokenizer, self.model = None, None

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, clean_up_tokenization_spaces=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **self.model_kwargs)
        self.model = torch.compile(self.model)

    def predict(self, prompt, stream=True, **inf_kwargs):
        if not self.model:
            self.load_model()
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            generated_ids = self.model.generate(**inputs,
                                                streamer=TextStreamer(self.tokenizer) if stream else None,
                                                **inf_kwargs).to("cuda")
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


if __name__ == "__main__":
    gpu = rh.cluster(name="rh-4-a10x", instance_type="A100:4")
    remote_hf_chat_model = HFChatModel(model_id="mistralai/Mixtral-8x7B",
                                       load_in_8bit=True,
                                       torch_dtype=torch.bfloat16).get_or_to(gpu, name="mixtral")

    test_prompt = "Why do Machine Learning Engineers let their infra push them around?"
    test_output = remote_hf_chat_model.predict(test_prompt, temperature=0.7, max_new_tokens=1000,
                                               repetition_penalty=1.0, do_sample=True)

    print("\n\n... Test Output ...\n")
    print(test_output)
