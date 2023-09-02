import argparse
import logging
import sys
import time
import torch
from sockeye.model import load_models
from . import inference
from . import arguments
params = arguments.ConfigArgumentParser(description='Translate CLI')
arguments.add_translate_cli_args(params)
args = params.parse_args()


def make_input(sen_id, line):
    return inference.make_input_from_plain_string(sen_id, line)
def translate(line,target):
    device=torch.device("cuda:"+str(args.device_id))
    tic = time.time()
    lines = line.split('\t\t')    
    targets = target.split('\t\t')
    models, source_vocabs, target_vocabs = load_models(device=device,
                                                       model_folders=args.models,
                                                       checkpoints=args.checkpoints,
                                                       dtype=args.dtype,
                                                       clamp_to_dtype=args.clamp_to_dtype,
                                                       inference_only=True,
                                                       knn_index=args.knn_index)
    for model in models:
        model.eval()
    
    brevity_penalty_weight = args.brevity_penalty_weight
    scorer = inference.CandidateScorer(
        length_penalty_alpha=args.length_penalty_alpha,
        length_penalty_beta=args.length_penalty_beta,
        brevity_penalty_weight=brevity_penalty_weight)
    scorer.to(models[0].dtype)
    translator = inference.Translator(
            ensemble_mode=args.ensemble_mode,
            scorer=scorer,
            batch_size=1,
            beam_search_stop=args.beam_search_stop,
            models=models,
            source_vocabs=source_vocabs,
            target_vocabs=target_vocabs,
            beam_size=1,
            device=device,
            force_decode=True,
            constant_length_ratio=-1.0
            )
    inputs = []
    for i, line in enumerate(lines):
        trans_input = make_input(i, line)
        inputs.append(trans_input)
    # 强制解码
    trans_output,hidden_states = translator.translate(inputs,targets)
    # 正常解码
    # trans_output = translator.translate(inputs)
    for output in trans_output:
        print(output.translation)
    # return "result"

def main():
    print(args)
    source_sentences = ["35 423 3894 24 104 386 17 1719 3114 9",
                        "35 921 5424 163 38 11 35 72 306 24 104 416 17 2548 1038 14 322 8795 8 386 24 17 887 121 122 27123 14 46011 19 104 386 9",
                        "99 302 58 2641 144 17 342 14 46011 24 80 5547 96 10577 57 4 9978 9",
                        "822 1343 24 91 33 1536 19 38953 2166 8055 7 4 4389 147 19 4 386 17 714 89 457 919 9",
                        "8769 4 1245 7723 7 38 81 29 16071 39 4 38953 2166 318 91 33 68 2433 19 38 9"]
    target_sentences = ["12 13 14 15 16 17 18",
                        "36 319 982 520 20 6 36 90 1180 6 189 5 654 812 70 1909 5 4107 6 114 554 1586 132 56 189 5 4107 1517 20 73 3996 1592 10",
                        "156 1141 73 1230 6 460 70 73 10495 3415 6 208 1267 105 13526 9394 10",
                        "252 807 6 136 189 1236 2512 13791 2326 6 4107 5 326 259 1797 28 89 218 16559 10",
                        "4160 11756 17769 6 136 189 208 3793 6 182 60 197 2512 13791 242 17769 10"       
    ]
    res = translate("\t\t".join(source_sentences),"\t\t".join(target_sentences))
    print(res)


if __name__=="__main__":
    print("hello")
    main()
