import codecs
import subprocess
import x2ms_adapter


def rounder(num):
    return round(num, 2)


def eval_distinct_file(run_file, ref_file, tokenizer=None, detokenizer=None):
    hypotheses = []
    with codecs.open(run_file, encoding='utf-8') as f:

        for line in f:
            temp = x2ms_adapter.tensor_api.split(line.strip('\n').strip('\r'), '\t', 3)
            assert len(temp) == 4
            hypotheses.append(x2ms_adapter.tensor_api.split(temp[3], ' ')) #assume it's already tokenized as it is for the groundtruth


    hypothesis_pipe = '\n'.join([' '.join(hyp) for hyp in hypotheses])
    pipe = subprocess.Popen(["perl", './evaluation/diversity.pl.remove_extension'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    pipe.stdin.write(hypothesis_pipe.encode())
    pipe.stdin.close()
    diversity = pipe.stdout.read()
    diversity = x2ms_adapter.tensor_api.split(str(diversity).strip())
    diver_uni = float(diversity[0][2:])
    diver_bi = float(diversity[1][:-3])

    return {'Distinct-1': rounder(diver_uni*100), 'Distinct-2': rounder(diver_bi*100)}


if __name__ == '__main__':
    eval_distinct_file("test_49.txt", None, None, None)