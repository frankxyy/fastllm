# 模型创建
from tools.fastllm_pytools import llm
import time

# model = llm.model("7b_fp16.flm")
model = llm.model("13b_fp16_v2.flm")
# model = llm.model("chatglm2-6b-fp16.flm")

# 生成回复
# print(model.response("你好"))

query = "你是谁"
system = "现在你是第四范式科技公司开发的多模态大规模预训练模型“小式”，你具有绘画、图片理解、自然语言处理、自动表格绘制、自动代码生成等等功能，请记住你的身份。"
prompt = ""
prompt += "USER：{} ASSISTANT：".format(query)
prompt = system + prompt

# 流式生成回复
t0 = time.time()
tot_time = 0
cnt = 0
text = ""
tot_iter = 1

for i in range(tot_iter):
    inner_cnt = 0
    inner_limit = 50
    for response in model.stream_response(prompt):
        t1 = time.time()
        print(response, flush = True)
        print("inner_cnt {}, token gen time: {}s".format(inner_cnt, t1-t0))

        if i>=tot_iter-1:
            tot_time += (t1 - t0)
            cnt += 1
            if response != '':
                text += response

        t0 = time.time()

        inner_cnt += 1
        if inner_cnt >= inner_limit:
            break
        

print('ave time per token is {}'.format(tot_time / cnt))
print("text = {}".format(text))

