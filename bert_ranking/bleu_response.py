import argparse
import json
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.tokenize import TweetTokenizer
import re
import os
import time
import subprocess

punctuation = '!,;:?"\''
filePath = '/data3/private/fengtao/Projects/cqr_t5/t5_response/predict_out/'


def readname(filePath):
    names = os.listdir(filePath)
    return names

def calc_nist_bleu_muilti(args):
    file_list = readname(filePath)
    for file in file_list:
        f = open(filePath + file, 'r')  # jsonlines
        all_lines = f.readlines()
        f.close()

        flag = True
        depth = 2
        max_depth = 0
        while flag:
            hyp0_file = open("hyp0.txt", "w", encoding="UTF-8")
            hyp1_file = open("hyp1.txt", "w", encoding="UTF-8")
            ref_file = open("ref.txt", "w", encoding="UTF-8")

            if not args.depth_analysis:
                flag = False

            count = 0
            for line in all_lines:
                example = json.loads(line)

                qn = int(example["query_number"])
                # print(qn)
                if qn > max_depth:
                    max_depth = qn
                if (args.depth_analysis) and qn != depth:
                    continue
                count += 1

                raw = example["input"][-1]
                output = example.get("omit_output", "") if args.reversed else example.get("output", "")
                output = output.strip()
                target = example["target"]

                raw = clean_str(raw)
                output = clean_str(output)
                target = clean_str(target)

                if args.reversed:
                    pass
                else:
                    ref_file.write(target + "\n")
                    hyp0_file.write(output + "\n")
                    hyp1_file.write(raw + "\n")

            hyp0_file.close()
            hyp1_file.close()
            ref_file.close()

            nist, sbleu_0 = _calc_nist_bleu(["ref.txt"], "hyp0.txt", 'temp', None)
            nist, sbleu_1 = _calc_nist_bleu(["ref.txt"], "hyp1.txt", 'temp', None)
            print('--------------------------')
            print(file)
            print(f"depth: {depth}, count: {count}, raw: {sbleu_1[1]}, output: {sbleu_0[1]}")

            if depth == max_depth:
                flag = False

            depth += 1


def calc_nltk_bleu_muilti(args):
    file_list = readname(filePath)
    for file in file_list:

        chencherry = SmoothingFunction()
        with open(filePath + file, 'r', encoding='UTF-8') as f:
            unchanged_bleu, output_bleu = 0, 0
            ln = 0
            for line in f:
                line = line[:-1] if line[-1] == '\n' else line
                example = json.loads(line)
                raw = example["input"][-1]
                output = example.get("omit_output", "") if args.reversed else example.get("output", "")
                output = output.strip()
                target = example["target"]

                raw = remove_punc(raw).split()
                output = remove_punc(output).split()
                target = remove_punc(target).split()

                if args.reversed:
                    unchanged_bleu += sentence_bleu([raw], target, weights=(0, 1, 0, 0),
                                                    smoothing_function=chencherry.method1)
                    output_bleu += sentence_bleu([raw], output, weights=(0, 1, 0, 0),
                                                 smoothing_function=chencherry.method1)
                else:
                    unchanged_bleu += sentence_bleu([target], raw, weights=(0, 1, 0, 0),
                                                    smoothing_function=chencherry.method1)
                    output_bleu += sentence_bleu([target], output, weights=(0, 1, 0, 0),
                                                 smoothing_function=chencherry.method1)
                ln += 1

        if args.reversed:
            print(f"target: {unchanged_bleu / ln}, output: {output_bleu / ln}")
        else:
            print('----------------')
            print(file)
            print(f"raw: {unchanged_bleu / ln}, output: {output_bleu / ln}")


def remove_punc(text: str):
    text = re.sub(r'[{}]+'.format(punctuation), '', text)
    return text.strip().lower()


def clean_str(txt, lang='en'):
    assert (lang in ['en', 'fr'])

    txt = txt.lower()
    txt = re.sub('eos', 'EOS', txt)
    for c in '«»“”':
        txt = re.sub(c, '"', txt)
    txt = re.sub('^', ' ', txt)
    txt = re.sub('$', ' ', txt)

    # url and tag
    words = []
    for word in txt.split():
        i = word.find('http')
        if i >= 0:
            word = word[:i] + ' ' + '__url__'
        if word.startswith('@') and word.endswith('@'):
            words.append('__tag__')
        else:
            words.append(word.strip())
    txt = ' '.join(words)

    # remove markdown URL
    txt = re.sub(r'\[([^\]]*)\] \( *__url__ *\)', r'\1', txt)

    # remove illegal char
    txt = re.sub('__url__', 'URL', txt)
    txt = re.sub('__tag__', 'TAG', txt)
    txt = re.sub(r"[^A-Za-zÀ-ÿ0-9():,.!?\"\']", " ", txt)
    txt = re.sub('URL', '__url__', txt)
    txt = re.sub('TAG', '__tag__', txt)

    # contraction
    tokenizer = TweetTokenizer(preserve_case=True)  # already lowercased but want to maintain, e.g., _EOS_
    if lang == 'en':
        txt = ' ' + ' '.join(tokenizer.tokenize(txt)) + ' '
        add_space = ["'s", "'m", "'re", "n't", "'ll", "'ve", "'d", "'em"]
        txt = txt.replace(" won't ", " will n't ")
        txt = txt.replace(" can't ", " can n't ")
        for a in add_space:
            txt = txt.replace(a + ' ', ' ' + a + ' ')
    elif lang == 'fr':
        ww = []
        for w in tokenizer.tokenize(txt):
            if "'" in w:
                ss = w.split("'")
                ww += [s + "'" for s in ss[:-1]] + [ss[-1]]
            else:
                ww.append(w)
        txt = ' '.join(ww)

    return ' '.join(txt.split())  # remove extra space


def _write_xml(paths_in, path_out, role, n_lines=None):
    # prepare .xml files for mteval-v14c.pl (calc_nist_bleu)
    # role = 'src', 'hyp' or 'ref'

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE mteval SYSTEM "">',
        '<!-- generated by https://github.com/golsun/NLP-tools -->',
        '<!-- from: %s -->' % paths_in,
        '<!-- as inputs for ftp://jaguar.ncsl.nist.gov/mt/resources/mteval-v14c.pl -->',
        '<mteval>',
    ]

    for i_in, path_in in enumerate(paths_in):

        # header ----

        if role == 'src':
            lines.append('<srcset setid="unnamed" srclang="src">')
            set_ending = '</srcset>'
        elif role == 'hyp':
            lines.append('<tstset setid="unnamed" srclang="src" trglang="tgt" sysid="unnamed">')
            set_ending = '</tstset>'
        elif role == 'ref':
            lines.append('<refset setid="unnamed" srclang="src" trglang="tgt" refid="ref%i">' % i_in)
            set_ending = '</refset>'

        lines.append('<doc docid="unnamed" genre="unnamed">')

        # body -----

        if role == 'src':
            body = [''] * n_lines
        else:
            with open(path_in, 'r', encoding='utf-8') as f:
                body = f.readlines()
            if n_lines is not None:
                body = body[:n_lines]
        for i in range(len(body)):
            line = body[i].strip('\n')
            line = line.replace('&', ' ').replace('<', ' ')  # remove illegal xml char
            if len(line) == 0:
                line = '__empty__'
            lines.append('<p><seg id="%i"> %s </seg></p>' % (i + 1, line))

        # ending -----

        lines.append('</doc>')
        if role == 'src':
            lines.append('</srcset>')
        elif role == 'hyp':
            lines.append('</tstset>')
        elif role == 'ref':
            lines.append('</refset>')

    lines.append('</mteval>')
    with open(path_out, 'w', encoding='utf-8') as f:
        f.write(str('\n'.join(lines)))


def _calc_nist_bleu(path_refs, path_hyp, fld_out='temp', n_lines=None):
    # call mteval-v14c.pl
    # ftp://jaguar.ncsl.nist.gov/mt/resources/mteval-v14c.pl
    # you may need to cpan install XML:Twig Sort:Naturally String:Util
    def makedirs(fld):
        if not os.path.exists(fld):
            os.makedirs(fld)

    makedirs(fld_out)

    if n_lines is None:
        n_lines = len(open(path_hyp, encoding='utf-8').readlines())
    _write_xml([''], fld_out + '/src.xml', 'src', n_lines=n_lines)
    _write_xml([path_hyp], fld_out + '/hyp.xml', 'hyp', n_lines=n_lines)
    _write_xml(path_refs, fld_out + '/ref.xml', 'ref', n_lines=n_lines)

    time.sleep(1)
    cmd = [
        'perl', 'mteval-v14c.pl',
        '-s', '%s/src.xml' % fld_out,
        '-t', '%s/hyp.xml' % fld_out,
        '-r', '%s/ref.xml' % fld_out,
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    output, error = process.communicate()

    lines = output.decode().split('\n')
    try:
        nist = lines[-6].strip('\r').split()[1:5]
        bleu = lines[-4].strip('\r').split()[1:5]
        return [float(x) for x in nist], [float(x) for x in bleu]

    except Exception:
        print('mteval-v14c.pl returns unexpected message')
        print('cmd = ' + str(cmd))
        print(output.decode())
        print(error.decode())
        return [-1] * 4, [-1] * 4


def calc_nltk_bleu(args):
    chencherry = SmoothingFunction()
    with open(args.prediction_file, 'r', encoding='UTF-8') as f:
        unchanged_bleu, output_bleu = 0, 0
        ln = 0
        for line in f:
            line = line[:-1] if line[-1] == '\n' else line
            example = json.loads(line)
            raw = example["input"][-1]
            output = example.get("omit_output", "") if args.reversed else example.get("output", "")
            output = output.strip()
            target = example["target"]

            raw = remove_punc(raw).split()
            output = remove_punc(output).split()
            target = remove_punc(target).split()

            if args.reversed:
                unchanged_bleu += sentence_bleu([raw], target, weights=(0, 1, 0, 0),
                                                smoothing_function=chencherry.method1)
                output_bleu += sentence_bleu([raw], output, weights=(0, 1, 0, 0), smoothing_function=chencherry.method1)
            else:
                unchanged_bleu += sentence_bleu([target], raw, weights=(0, 1, 0, 0),
                                                smoothing_function=chencherry.method1)
                output_bleu += sentence_bleu([target], output, weights=(0, 1, 0, 0),
                                             smoothing_function=chencherry.method1)
            ln += 1

    if args.reversed:
        print(f"target: {unchanged_bleu / ln}, output: {output_bleu / ln}")
    else:
        print(f"raw: {unchanged_bleu / ln}, output: {output_bleu / ln}")


def calc_nist_bleu(args):
    f = open(args.prediction_file, 'r')  # jsonlines
    all_lines = f.readlines()
    f.close()

    flag = True
    depth = 2
    max_depth = 0
    while flag:
        hyp0_file = open("hyp0.txt", "w", encoding="UTF-8")
        hyp1_file = open("hyp1.txt", "w", encoding="UTF-8")
        ref_file = open("ref.txt", "w", encoding="UTF-8")

        if not args.depth_analysis:
            flag = False

        count = 0
        for line in all_lines:
            example = json.loads(line)

            qn = int(example["query_number"])
            # print(qn)
            if qn > max_depth:
                max_depth = qn
            if (args.depth_analysis) and qn != depth:
                continue
            count += 1

            raw = example["input"][-1]
            output = example.get("omit_output", "") if args.reversed else example.get("output", "")
            output = output.strip()
            target = example["target"]

            raw = clean_str(raw)
            output = clean_str(output)
            target = clean_str(target)

            if args.reversed:
                pass
            else:
                ref_file.write(target + "\n")
                hyp0_file.write(output + "\n")
                hyp1_file.write(raw + "\n")

        hyp0_file.close()
        hyp1_file.close()
        ref_file.close()

        nist, sbleu_0 = _calc_nist_bleu(["ref.txt"], "hyp0.txt", 'temp', None)
        nist, sbleu_1 = _calc_nist_bleu(["ref.txt"], "hyp1.txt", 'temp', None)

        print(f"depth: {depth}, count: {count}, raw: {sbleu_1[1]}, output: {sbleu_0[1]}")

        if depth == max_depth:
            flag = False

        depth += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", default=None, type=str, help="NDJson")
    parser.add_argument("--reversed", default=None, action='store_true', help="reversed GPT-2 or not")
    parser.add_argument("--nltk", action="store_true", help="Use nltk to measure bleu")
    parser.add_argument("--depth_analysis", action="store_true", help="Whether to analysis depth")
    # parser.add_argument("--raw_or_output", default=None, type=str, required=True, help="Choose to compute raw or output.")
    args = parser.parse_args()


    calc_nist_bleu_muilti(args)



if __name__ == "__main__":
    main()