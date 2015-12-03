import pyvw

valid_labels = {'n.act': 0,
'n.animal': 1,
'n.artifact': 2,
'n.attribute': 3,
'n.body': 4,
'n.cognition': 5,
'n.communication': 6,
'n.event': 7,
'n.feeling': 8,
'n.food': 9,
'n.group': 10,
'n.location': 11,
'n.motive': 12,
'n.natural_object': 13,
'n.other': 14,
'n.person': 15,
'n.phenomenon': 16,
'n.plant': 17,
'n.possession': 18,
'n.process': 19,
'n.quantity': 20,
'n.relation': 21,
'n.shape': 22,
'n.state': 23,
'n.substance': 24,
'n.time': 25,
'v.body': 26,
'v.change': 27,
'v.cognition': 28,
'v.communication': 29,
'v.competition': 30,
'v.consumption': 31,
'v.contact': 32,
'v.creation': 33,
'v.emotion': 34,
'v.motion': 35,
'v.perception': 36,
'v.possession': 37,
'v.social': 38,
'v.stative': 39,
'v.weather': 40}
valid_labels_rev = { v:k for k,v in valid_labels.iteritems() }

class BIO:
    # construct a BIO object using a bio type ('O', 'B' or 'I') and a
    # optionally a label (that can be used to capture the supersense tag). 
    # this additionally computes a numeric_label to be used by vw
    def __init__(self, bio, label=None):
        if bio != 'O' and bio != 'B' and bio != 'I':
            raise TypeError
        self.bio = bio
        self.label = None   # the label will only be needed for supersenses
        self.numeric_label = 1
        if self.bio == 'B':
            self.numeric_label = 2 
        elif self.bio == 'I':
            self.numeric_label = 3 

    # a.can_follow(b) returns true if:
    #    a is O and b is I or O or
    #    a is B and b is I or O or
    #    ...
    def can_follow(self, prev):
        return (self.bio == 'O' and (prev.bio == 'I' or prev.bio == 'O') ) or \
               (self.bio == 'B' and (prev.bio == 'I' or prev.bio == 'O') ) or \
               (self.bio == 'I' and (prev.bio == 'B' or prev.bio == 'I') ) 

    # given a label, produce a list of all valid BIO items that can
    # come next. 
    def valid_next(self):
        valid = [] #TODO
        return valid 

    # produce a human-readable string
    def __str__( self): return self.bio #return 'O' if self.bio == 'O' else (self.bio + '-' + self.label)
    def __repr__(self): return self.__str__()

    # compute equality
    def __eq__(self, other):
        if not isinstance(other, BIO): return False
        return self.bio == other.bio and self.label == other.label
    def __ne__(self, other): return not self.__eq__(other)
        
# convert a numerical prediction back to a BIO label
def numeric_label_to_BIO(num):
    if not isinstance(num, int):
        raise TypeError
    if num == 1:
        return BIO('O')
    elif num == 2:
        return BIO('B')
    elif num == 3:
        return BIO('I')

# given a previous PREDICTED label (prev), which may be incorrect; and
# the current TRUE label (truth), generate a list of valid reference
# actions. the return type should be [BIO]. for example, if the truth
# is O or B, then regardless of what prev is the correct thing to do
# is [truth]. the most important thing is to handle the case when, for
# instance, truth is I but prev is neither I nor B
def compute_reference(prev, truth):
    return [ truth ]  # TODO

        
class MWE(pyvw.SearchTask):
    def __init__(self, vw, sch, num_actions):
        # you must must must initialize the parent class
        # this will automatically store self.sch <- sch, self.vw <- vw
        pyvw.SearchTask.__init__(self, vw, sch, num_actions)
        
        # for now we will use AUTO_HAMMING_LOSS; in Part II, you should remove this and implement a more task-focused loss
        # like one-minus-F-measure.
        sch.set_options( sch.AUTO_HAMMING_LOSS | sch.AUTO_CONDITION_FEATURES )

    def _run(self, sentence):
        output = []
        prev   = BIO('O')   # store the previous prediction
        for n in range(len(sentence)):
            # label is a BIO, word is a string and pos is a string
            label,word,lemma,pos = sentence[n]

            with self.make_example(word, lemma, pos) as ex:  # construct the VW example
                # first, compute the numeric labels for all valid reference actions
                refs  = [ bio.numeric_label for bio in compute_reference(prev, label) ]
                # next, because some actions are invalid based on the
                # previous decision, we need to compute a list of
                # valid actions available at this point
                valid = [ bio.numeric_label for bio in prev.valid_next() ]
                # make a prediction
                pred  = self.sch.predict(examples   = ex,
                                         my_tag     = n+1,
                                         oracle     = refs,
                                         condition  = [(n, 'p'), (n-1, 'q')],
                                         allowed    = valid)
                # map that prediction back to a BIO label
                this  = numeric_label_to_BIO(pred)
                # append it to output
                output.append(this)
                # update the 'previous' prediction to the current
                prev  = this

        # return the list of predictions as BIO labels
        return output

    def make_example(self, word, lemma, pos):
        return self.example({
            'w': [word],
            'l': [lemma],
            'p': [pos],
        })


def make_data(BIO,filename):
    data = []
    sentence = []
    f = open(filename,'r')
    for l in f:
        l = l.strip()
        # at end of sentence
        if l == "":
            data.append(sentence)
            sentence = []
        else:
            [offset,word,lemma,pos,mwe,parent,strength,ssense,sid] = l.split('\t')
            sentence.append((BIO(mwe),word,lemma,pos))
    return data



if __name__ == "__main__":
    # input/output files
    trainfilename='dimsum16.p3.train.contiguous'
    testfilename='dimsum16.p3.test.contiguous'
    outfilename='dimsum16.p3.test.contiguous.out'

    # read in some examples to be used as training/dev set
    train_data = make_data(BIO,trainfilename)

    # initialize VW and sequence labeler as learning to search
    vw = pyvw.vw(search=9, quiet=True, search_task='hook', ring_size=1024, \
                 search_rollin='learn', search_rollout='none')

    # tell VW to construct your search task object
    sequenceLabeler = vw.init_search_task(MWE)

    # train!
    # we make 5 passes over the training data, training on the first 80%
    # examples (we retain the last 20% as development data)
    print 'training!'
    N = int(0.8 * len(train_data))
    for i in xrange(5):
        print 'iteration ', i, ' ...'
        sequenceLabeler.learn(train_data[0:N])
        
    # now see the predictions on 20% held-out sentences 
    print 'predicting!' 
    hamming_loss, total_words = 0,0
    for n in range(N, len(train_data)):
        truth = [label for label,word,lemma,pos in train_data[n]]
        pred  = sequenceLabeler.predict( [(BIO('O'),word,lemma,pos) for label,word,lemma,pos in train_data[n]] )
        for i,t in enumerate(truth):
            if t != pred[i]:
                hamming_loss += 1
            total_words += 1
    #    print 'predicted:', '\t'.join(map(str, pred))
    #    print '    truth:', '\t'.join(map(str, truth))
    #    print ''
    print 'total hamming loss on dev set:', hamming_loss, '/', total_words

    # In Part II, you will have to output predictions on the test set.
    #test_data = make_data(BIO,testfilename)
    #for n in range(N, len(test_data)):
        # make predictions for current sentence
        #pred  = sequenceLabeler.predict( [(BIO('O'),word,lemma,pos) for label,word,lemma,pos in train_data[n]] )


