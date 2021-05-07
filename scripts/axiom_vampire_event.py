from nltk import *

from nltk.sem import Expression
from nltk.sem.logic import *
lexpr = Expression.fromstring
def vampire_axioms(adjdic, antonyms, Objs, tVerbs, iVerbs, predicates, lst):
    axiom = []
    types = []
#     print("predicates",predicates)
    for test_ind, pred in enumerate(predicates):
        if pred[0][0] == '_':
            pred[0] = pred[0][1:]

        if (pred[0] == 'many') and not (pred[0] in adjdic):
            adjdic[pred[0]] = 'POS'
        if (pred[0] == 'few') and not (pred[0] in adjdic):
            adjdic[pred[0]] = 'NEG'

        if '$less' in pred[0]:
            deg = lexpr('$less(0,_d0)')
            axiom.append(deg)
            many_type = 'tff(many_type, type , many : $i * $int > $o).'
            types.append(many_type)
            ax1 = lexpr('all x d1. (many(x,d1) -> all d2. ($lesseq(d2,d1) -> many(x,d2)))')
            axiom.append(ax1)

        # Adjectives
        if pred[0] in adjdic:
#             print("===cases===", test_ind,pred[1])
            if len(pred[1]) >= 2 and '_d0' in pred[1][1]:
                deg = lexpr('$less(0,_d0)')
                axiom.append(deg)
            if 'person' in Objs:
                if '_np' in pred[1][1]:
                    defcom = lexpr('all x. (' + pred[0] +
                                   '(x,_np(_u,_th(_u))) <-> ' + pred[0] +
                                   '(x,_np(_person,_th(_person))))')
                else:
                    defcom = lexpr('all x. (' + pred[0] + '(x,_th(_u)) \
                    <-> ' + pred[0] + '(x,_th(_person)))')
                axiom.append(defcom)

            if '_np' in pred[1][1]:
                np = lexpr('all x d1 d2.($lesseq(d1,d2) <-> $lesseq(_np(x,d1),_np(x,d2)))')
                axiom.append(np)

            # Positive adjectives
            if antonyms != [] or '_th(' not in ((pred[1])[1]):
                if adjdic[pred[0]][:3] == 'POS':
                    if '$' in pred[1][1]:
                        upper = lexpr('all x. exists d1.(' + pred[0] +
                                      '(x,d1) & -exists d2.($less(d1,d2) & ' +
                                      pred[0] + '(x,d2)))')
                        axiom.append(upper)
                    if pred[0] != 'many' or (pred[1][1])[0] != 'd':
                        ax1 = lexpr('all x d1. (' + pred[0] +
                                    '(x,d1) -> all d2. ($lesseq(d2,d1) -> '
                                    + pred[0] + '(x,d2)))')
                        axiom.append(ax1)

            # Negative adjectives
                elif adjdic[pred[0]][:3] == 'NEG':
                    if '$' in pred[1][1]:
                        lower = lexpr('all x. exists d1.(' + pred[0] +
                                      '(x,d1) & -exists d2.($less(d2,d1) & '
                                      + pred[0] + '(x,d2)))')
                        axiom.append(lower)

                    if pred[0] != 'few' or (pred[1][1])[0] != 'd':
                        ax2 = lexpr('all x d1. (' + pred[0] +
                                    '(x,d1) -> all d2. ($lesseq(d1,d2) -> '
                                    + pred[0] + '(x,d2)))')
                        axiom.append(ax2)
                else:
                    pass

        elif pred[0] == 'AccI':
            att1 = lexpr('all e. (AccI(e,' + pred[1][1] +
                         ') -> (_know(e) ->' + pred[1][1] + '))')
            att2 = lexpr('all e. (AccI(e,' + pred[1][1] +
                         ') -> (_forget(e) ->' + pred[1][1] + '))')
            att3 = lexpr('all e. (AccI(e,' + pred[1][1] +
                         ') -> (_learn(e) ->' + pred[1][1] + '))')
            att4 = lexpr('all e. (AccI(e,' + pred[1][1]
                         + ') -> (_remember(e) ->' + pred[1][1] + '))')
            att5 = lexpr('all e. (AccI(e,' + pred[1][1] +
                         ') -> (_manage(e) ->' + pred[1][1] + '))')
            att6 = lexpr('all e. (AccI(e,' + pred[1][1] +
                         ') -> (_fail(e) -> -' + pred[1][1] + '))')
            axiom.extend([att1, att2, att3, att4, att5, att6])

        # former
        elif (('former' in pred[0]) and (type(pred[1][0]) is str)):
            aff = lexpr('all x. (_former(' + pred[1][0] +
                        ') -> -' + pred[1][0] + ')')
            axiom.append(aff)

        elif pred[0] == 'true':
            tr = lexpr('_true(' + pred[1][0] + ') -> ' + pred[1][0])
            axiom.append(tr)

        elif pred[0] == 'false':
            fl = lexpr('_false(' + pred[1][0] + ') -> -' + pred[1][0])
            axiom.append(fl)

        else:
            pass

    if antonyms != []:
        for antonym in antonyms:
            Fp = antonym[0]
            Fm = antonym[1]
            ax3 = lexpr('all x d.(' + Fp + '(x,d) <-> -' + Fm + '(x,d))')
            axiom.append(ax3)

    if lst != []:
        for i in range(len(Objs)):
            for j in range(len(Objs)):
                if not Objs[i] == Objs[j]:
                    ax = lexpr('(all x. (' + Objs[i] +
                               '(x) <-> ' + Objs[j] +
                               '(x))) <-> (_th(' + Objs[i] + ') = _th('
                               + Objs[j] + '))')
                    axiom.append(ax)

    if tVerbs != []:
        for verb in tVerbs:
            verbax = lexpr('all e1 e2.(' + verb + '(e1) & ' + verb + '(e2)  & (subj(e1) = subj(e2)) & (acc(e1) = acc(e2)) -> (e1 = e2))')
            if verbax not in axiom:
                axiom.append(verbax)
    if iVerbs != []:
        for verb in iVerbs:
            verbax = lexpr('all e1 e2.(' + verb + '(e1) & ' + verb +
                           '(e2) & (subj(e1) = subj(e2)) -> (e1 = e2))')
            if verbax not in axiom:
                axiom.append(verbax)

    axiom = set(axiom)
    axiom = list(axiom)
    return types, axiom


def main():
    adjdic = {}
    antonyms = []
    Objs = []
    lst = []
    predicates = []
    axioms, lemma = vampire_axioms(adjdic, antonyms, Objs, predicates, lst)
    print(axioms)


if __name__ == "__main__":
    main()
