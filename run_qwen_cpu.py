from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM
model_name = "/mnt/data/Qwen-7B-Chat" #本地路径
prompt = "领导：你这是什么意思？小明：没什么意思。意思意思。领导：你这就不够意思了。小明：小意思，小意思。领导：你这人真有意思。小明：其实也没有别的意思。领导：那我就不好意思了。小明：是我不好意思。请问：以上\"意思\" 分别是什么意思。"
tokenizer = AutoTokenizer .from_pretrained (
 model_name,
trust_remote_code=True
)
model = AutoModelForCausalLM .from_pretrained (
 model_name,
trust_remote_code=True,
torch_dtype="auto" # 自动选择float32/float16（根据模型配置）
).eval ()
inputs = tokenizer (prompt, return_tensors="pt").input_ids
streamer = TextStreamer (tokenizer)
outputs = model .generate (inputs, streamer=streamer, max_new_tokens=300)
