
from graphviz import Digraph


class MyCollector:
    
    @staticmethod
    def qwen3_2_png():

        dot = Digraph(comment="Qwen3 + LoRA Summary")

        dot.node("embed", "Embedding\n(151936 → 2560)")
        dot.node("q_proj", "q_proj\nLinear (2560→4096)\nLoRA (2560→32→4096)")
        dot.node("k_proj", "k_proj\nLinear (2560→1024)\nLoRA (2560→32→1024)")
        dot.node("v_proj", "v_proj\nLinear (2560→1024)\nLoRA (2560→32→1024)")
        dot.node("o_proj", "o_proj\nLinear (4096→2560)\nLoRA (4096→32→2560)")
        dot.node("gate_proj", "gate_proj\nLinear (2560→9728)\nLoRA (2560→32→9728)")
        dot.node("up_proj", "up_proj\nLinear (2560→9728)\nLoRA (2560→32→9728)")
        dot.node("down_proj", "down_proj\nLinear (9728→2560)\nLoRA (9728→32→2560)")
        dot.node("input_ln", "Input RMSNorm\n(2560)")
        dot.node("post_attn_ln", "Post-Attn RMSNorm\n(2560)")
        dot.node("final_ln", "Final RMSNorm\n(2560)")
        dot.node("act_fn", "Activation: SiLU")
        dot.node("lm_head", "LM Head\nLinear (2560 → 151936)")

        dot.edges([
            ("embed", "input_ln"),
            ("input_ln", "q_proj"),
            ("input_ln", "k_proj"),
            ("input_ln", "v_proj"),
            ("q_proj", "o_proj"),
            ("k_proj", "o_proj"),
            ("v_proj", "o_proj"),
            ("o_proj", "post_attn_ln"),
            ("post_attn_ln", "gate_proj"),
            ("post_attn_ln", "up_proj"),
            ("gate_proj", "down_proj"),
            ("up_proj", "down_proj"),
            ("down_proj", "act_fn"),
            ("act_fn", "final_ln"),
            ("final_ln", "lm_head")
        ])

        # 输出为 PNG 图像
        dot.render("qwen3_lora_model", format="png", cleanup=True)

    @staticmethod
    def fp(a):
        
        print('----------------------------------------')
        print(a)
        print('----------------------------------------\n\n')

if __name__ == '__main__':

    collector = MyCollector()
    collector.qwen3_2_png()
