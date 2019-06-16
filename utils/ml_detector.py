import os
import numpy


class EventDetectorForML():
    '''
    Detect blood_oxygen
    Method:
        1. window size: 20s. if crest - trough >= 3, then detecte.
        2. once detected, record its start time, and mark according to events.
        3. window move forward by half a window size
        calculate the hit ratio: non-zero marked devents len / events len
    '''

    def __init__(self, root=None, extend=False):
        '''
        initial
        '''
        if root is None:
            self.data_dir = '../data/train/'
        else:
            self.data_dir = root

        self.extend = extend

        self.window_size = 3 * 20
        self.error_length = 3 * 3
        self.to_skip_length = self.window_size
        self.devents = []  # list of [start_index, mark]
        self.fragments = []  # result, list of [fragment, mark]
        self.class_total = list(0. for i in range(3))

        self._init_subject_names()

    def _init_subject_names(self):
        '''
        extract patients name from file
        '''
        self.subject_names = []
        files = os.listdir(self.data_dir)
        if '.DS_Store' in files:  # macos
            files.remove('.DS_Store')
        for file in files:
            if file.count('_') > 0:
                subject_name = file.split('_')[0]
            else:
                subject_name = file.split('-')[0]
            if subject_name not in self.subject_names:
                self.subject_names.append(subject_name)

    def detect(self):
        '''
        Detect abnormal events in a data directory
        '''
        # self.subject_names = [self.subject_names[0]] # Trick: now one case
        for subject_name in self.subject_names:
            try:
                self.subject_name = subject_name
                self.read_data()
                self.formate_data()
                self.detect_one_subject()
                self.get_hit_ratio()
                self.append_fragments()
            except Exception as e:
                # if exists bad data
                print(subject_name)

    def read_data(self):
        '''
        read data from blood oxygen file and event file
        '''
        self.blood_oxygen = []
        self.events = []

        blood_oxygen_file = os.path.join(
            self.data_dir, self.subject_name + '_血氧_.txt')

        events_file = os.path.join(
            self.data_dir, self.subject_name + '-事件.txt')

        # detect abnormal event
        with open(blood_oxygen_file) as txt_file:
            for line in txt_file:
                record = line.split()
                record[1] = int(record[1])
                self.blood_oxygen.append(record)

        # mark the true label
        with open(events_file) as txt_file:
            for line in txt_file:
                record = line.split()
                record[1] = int(record[1])
                record[2] = int(record[2])
                self.events.append(record)

    def formate_data(self):
        '''
        Formate:
            remove microseconds
            padd events, length + error_length
        '''
        self.blood_oxygen = [
            [x[0][:x[0].find(':', 3)+3], x[1]] for x in self.blood_oxygen]
        self.events = [[x[0][:x[0].find(':', 3)+3], x[1], x[2]]
                       for x in self.events]

        self.dates = [x[0] for x in self.blood_oxygen]

        def pad_hour(x):
            if len(x) == 7:
                return '0' + x
            else:
                return x
        self.dates = list(map(pad_hour, self.dates))
        self.values = [x[1] for x in self.blood_oxygen]

        self.padded_event_dates = []
        self.padded_marks = []
        for event in self.events:
            index = self.dates.index(event[0])
            padding_length = 3 * event[2] + 3 * self.error_length
            self.padded_event_dates.extend(
                self.dates[index:index+padding_length])
            self.padded_marks.extend([event[1] for i in range(padding_length)])

        # print(self.dates[0])
        # print(self.dates[1800*24])
        # for x in zip(self.padded_event_dates[:500], self.padded_marks[:500]):
        #     print(x[0], x[1])

    def detect_one_subject(self):
        '''
        detect one file
        '''
        skip_count = self.to_skip_length
        for i in range(len(self.dates)):
            if skip_count < self.to_skip_length:
                skip_count += 1
                continue
            if self.is_abnormal(i):
                self.devents.append([i, self.mark(i)])
                skip_count = 0

    def is_abnormal(self, i):
        '''
        detect one suquence
        '''
        if i+self.window_size > len(self.values):  # overflow
            return False
        fragment = self.values[i:i+self.window_size]
        a, b = [max(fragment), min(fragment)]
        last_a_index = len(fragment) - 1 - fragment[::-1].index(a)
        if b < 60:  # mechanical error
            return False
        if last_a_index < 3 and a - b >= 2 and last_a_index < fragment.index(b):
            self.to_skip_length = fragment.index(b)
            return True
        else:
            return False

    def mark(self, i):
        '''
        get true label of one suquence
        Standard：

        '''
        date = self.dates[i]
        if date in self.padded_event_dates:
            return int(self.padded_marks[self.padded_event_dates.index(date)])
        else:
            return 0

    def get_hit_ratio(self):
        '''
        get accuracy of abnormal detection
        '''
        hit_count = 0
        for devent in self.devents:
            # print(self.dates[devent[0]], devent[1])
            if devent[1] > 0:
                hit_count = hit_count + 1

        # print(len(self.devents))
        # print(len(self.events))
        # print(hit_count)
        # print('hit ratio is: ' + str(hit_count / len(self.events)))
        # TODO: the devents can be close and maybe some of them are the
        # same event, it's okey. trim the non-zero close events after
        # prediction is ok.

    def append_fragments(self):
        '''
        approximate sampling
        add neighbor suquence to extend data
        '''
        # print(len(self.devents))
        for devent in self.devents:
            index = devent[0]
            mark = devent[1]
            self.class_total[mark] += 1
            self.fragments.append(
                [self.values[index:index + self.window_size], mark])
            if self.extend == True:
                self.extend_data(index, mark)
        self.devents = []

    def extend_data(self, index, mark):
        '''
        data enhance
        downsampling
        '''
        if index-10 >= 0:
            if mark == 2:
                self.class_total[mark] += 1
                self.fragments.append(
                    [self.values[index - 10:index - 10 + self.window_size], mark])
                # self.fragments.append(
                #     [self.values[index + 10:index + 10 + self.window_size], mark])
            if mark == 1:
                self.class_total[mark] += 1
                self.fragments.append(
                    [self.values[index - 10: index-10 + self.window_size], mark])


if __name__ == "__main__":
    d = EventDetectorForML(root="../data/train")
    d.detect()
