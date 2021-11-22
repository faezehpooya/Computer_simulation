from numpy import random as rn
import numpy
import math
import random
import time
import matplotlib.pyplot as plt


class PERSON:
    def __init__(self, has_corona, bored_time, arrival_time, service_time):
        self.has_corona = has_corona
        self.bored_time = bored_time
        self.arrival_time = arrival_time
        self.bored_reception_clock = arrival_time + bored_time
        self.respons_time=service_time
        # reception
        self.service_time = service_time
        self.end_time = None
        self.wait_in_reception_queue = None
        self.total_wait = None


class DOCTOR:
    def __init__(self, mean_service_rate):
        self.check_up_mean_service_rate = mean_service_rate
        self.cur_pat_type_corona = None
        self.finish_check_up_clock = -1


class ROOM:
    def __init__(self, room_num, number_of_doctors, mean_service_rates):

        self.Doctors = []
        for i in range(number_of_doctors):
            self.Doctors.append(DOCTOR(mean_service_rates[i]))
        self.corona_patients_queue = []
        self.normal_patients_queue = []
        self.number_of_corona_pat_in_room = 0
        self.number_of_normal_pat_in_room = 0
        self.room_is_full = False
        self.total_finished = 0
        ###############################################
        self.total_time_in_sys_corona_pats = 0
        self.total_time_in_sys_normal_pats = 0

        self.total_time_wait_in_room_q_corona = 0
        self.total_time_wait_in_room_q_normal = 0

    def check_up(self, clock):

        number_of_busy_doctors = 0
        number_of_pat_finished_check_up = 0

        for doctor in self.Doctors:

            if doctor.finish_check_up_clock > clock:
                number_of_busy_doctors += 1
                continue

            if doctor.finish_check_up_clock <= clock:
                if doctor.finish_check_up_clock == clock:
                    if doctor.cur_pat_type_corona:
                        self.number_of_corona_pat_in_room -= 1
                    else:
                        self.number_of_normal_pat_in_room -= 1
                    self.total_finished += 1
                    number_of_pat_finished_check_up += 1
                    number_of_busy_doctors -= 1

                if (len(self.corona_patients_queue) or len(self.normal_patients_queue)):

                    number_of_busy_doctors += 1
                    service_rate = int(rn.exponential(doctor.check_up_mean_service_rate)) + 1

                    doctor.finish_check_up_clock = clock + service_rate

                    if len(self.corona_patients_queue):
                        doctor.cur_pat_type_corona = True
                        patient = self.corona_patients_queue.pop(0)
                        self.number_of_corona_pat_in_room += 1
                        wait= clock - (
                                patient.arrival_time + patient.service_time + patient.wait_in_reception_queue)
                        self.total_time_wait_in_room_q_corona += wait
                        patient.total_wait=patient.wait_in_reception_queue+wait
                        self.total_time_in_sys_corona_pats += (clock + service_rate - patient.arrival_time)

                    else:
                        doctor.cur_pat_type_corona = False
                        patient = self.normal_patients_queue.pop(0)
                        self.number_of_normal_pat_in_room += 1
                        wait=clock - (
                                patient.arrival_time + patient.service_time + patient.wait_in_reception_queue)
                        self.total_time_wait_in_room_q_normal += wait
                        patient.total_wait=patient.wait_in_reception_queue+wait
                        self.total_time_in_sys_normal_pats += (clock + service_rate - patient.arrival_time)

                    patient.respons_time+=service_rate

        self.room_is_full = number_of_busy_doctors == len(self.Doctors)

        return number_of_pat_finished_check_up


class Reception_class:
    def __init__(self, persons_corona, persons_normal):
        self.Queue_to_room = []
        self.corona_patient_queue = persons_corona.copy()
        self.corona_pat_q_index = 0
        self.normal_patient_queue = persons_normal.copy()
        self.normal_pat_q_index = 0
        self.reception_clock_finish_service = -1
        self.reception_busy = False
        self.patient_in_reception = None

        self.total_time_wait_in_q_corona_pats = 0
        self.total_time_wait_in_q_normal_pats = 0

        self.time_tired = 0
        self.time_while = 0
        self.time_add = 0
        self.counter = 0

    def Reception(self, clock, number_of_patients):

        left_reception_corona_pats = 0
        left_reception_normal_pats = 0

        time2 = time.time()

        if clock >= self.reception_clock_finish_service:

            if clock == self.reception_clock_finish_service:
                self.reception_busy = False
                self.Queue_to_room.append(self.patient_in_reception)

            # if  patient is tired -- > left queue
            time2 = time.time()

            while True:
                if (len(self.corona_patient_queue) - self.corona_pat_q_index) and self.corona_patient_queue[
                    self.corona_pat_q_index].bored_reception_clock <= clock:
                    self.corona_pat_q_index += 1
                    number_of_patients -= 1
                    left_reception_corona_pats += 1
                else:
                    break

            while True:
                if (len(self.normal_patient_queue) - self.normal_pat_q_index) and self.normal_patient_queue[
                    self.normal_pat_q_index].bored_reception_clock <= clock:
                    self.normal_pat_q_index += 1
                    number_of_patients -= 1
                    left_reception_normal_pats += 1
                else:
                    break

            time3 = time.time()
            self.time_tired += (time3 - time2)

            # next patient come to reception

            if len(self.corona_patient_queue) - self.corona_pat_q_index and self.corona_patient_queue[
                self.corona_pat_q_index].arrival_time <= clock:
                self.patient_in_reception = self.corona_patient_queue[self.corona_pat_q_index]
                self.corona_pat_q_index += 1
                self.reception_busy = True
                self.reception_clock_finish_service = clock + self.patient_in_reception.service_time
                # wait_in_reception_queue = \
                self.patient_in_reception.wait_in_reception_queue = clock - self.patient_in_reception.arrival_time
                self.total_time_wait_in_q_corona_pats += self.patient_in_reception.wait_in_reception_queue

            elif len(self.normal_patient_queue) - self.normal_pat_q_index and self.normal_patient_queue[
                self.normal_pat_q_index].arrival_time <= clock:

                self.patient_in_reception = self.normal_patient_queue[self.normal_pat_q_index]
                self.normal_pat_q_index += 1
                self.reception_clock_finish_service = clock + self.patient_in_reception.service_time
                self.reception_busy = True
                self.patient_in_reception.wait_in_reception_queue = clock - self.patient_in_reception.arrival_time
                self.total_time_wait_in_q_normal_pats += self.patient_in_reception.wait_in_reception_queue

            time4 = time.time()
            self.time_add += time4 - time3

            len_reception_normal=0.9*clock/7 -self.normal_pat_q_index


        return number_of_patients, left_reception_corona_pats, left_reception_normal_pats


class hospital:
    def __init__(self, M, number_of_doctors_per_room, mean_check_up_time, persons_corona, persons_normal,corona_totals):
        self.number_of_patients = len(persons_corona) + len(persons_normal)
        self.persons_corona = persons_corona
        self.persons_normal = persons_normal

        self.reception = Reception_class(perosons_corona, perosons_normal)

        self.number_of_doctors_per_room = number_of_doctors_per_room
        self.mean_check_up_time = mean_check_up_time
        self.Rooms = []
        self.corona_totals = corona_totals

        self.time_reception_took = 0
        self.time_tired_room_took = 0
        self.time_room_took = 0
        self.time_calculate_min_len = 0

        self.time_calculation_in = 0
        self.time_calculation_in1 = 0
        self.time_calculation_in2 = 0


        self.reception_arrived_index_corona=0
        self.reception_arrived_index_normal=0
        

        for i in range(M):
            self.Rooms.append(ROOM(i, self.number_of_doctors_per_room[i], self.mean_check_up_time[i]))

    def start_simulation(self):

        clock = 0
        temp_counter = 0
        total_len_reception_queue=0
        len_reception_queue_corona = 0
        len_reception_queue_normal=0
        len_rooms_queue = [0 for i in range(len(self.Rooms))]
        dict_people_in_system = {}
        dict_people_in_system_corona = {}
        dict_people_in_system_normal = {}

        dict_people_in_queue = {}
        dict_people_in_queue_corona = {}
        dict_people_in_queue_normal = {}

        left_reception_normal_pats = 0
        left_reception_corona_pats = 0
        left_rooms_normal_pats = 0
        left_rooms_corona_pats = 0
        while self.number_of_patients >= 0:




            left_reception_normal_pats_temp = 0
            left_reception_corona_pats_temp = 0

            time1 = time.time()
            if len(self.reception.corona_patient_queue) or len(
                    self.reception.normal_patient_queue) or self.reception.patient_in_reception:

                while len(self.reception.normal_patient_queue)-self.reception_arrived_index_normal:
                    if clock >= self.reception.normal_patient_queue[self.reception_arrived_index_normal].arrival_time:
                        self.reception_arrived_index_normal+=1
                    else:
                        break
                #print(self.reception_arrived_index_normal,self.reception.normal_pat_q_index)
                len_reception_queue_normal=self.reception_arrived_index_normal-self.reception.normal_pat_q_index
                #print("normal",self.reception_arrived_index_normal-self.reception.normal_pat_q_index)

                while len(self.reception.corona_patient_queue)-self.reception_arrived_index_corona:
                    if clock>=self.reception.corona_patient_queue[self.reception_arrived_index_corona].arrival_time:
                        self.reception_arrived_index_corona+=1
                    else:
                        break
                len_reception_queue_corona=self.reception_arrived_index_corona-self.reception.corona_pat_q_index
                #print("corona",self.reception_arrived_index_corona-self.reception.corona_pat_q_index)


                self.number_of_patients, left_reception_corona_pats_temp, left_reception_normal_pats_temp = self.reception.Reception(
                    clock, self.number_of_patients)

            # add number of pat left system in reception q
            left_reception_normal_pats += left_reception_normal_pats_temp
            left_reception_corona_pats += left_reception_corona_pats_temp

            time2 = time.time()
            self.time_reception_took += time2 - time1
            # send patient to the shortest queue

            while len(self.reception.Queue_to_room):
                qu_len = [
                    len(self.Rooms[i].corona_patients_queue) + len(self.Rooms[i].normal_patients_queue) + self.Rooms[
                        i].room_is_full for i in range(M)]
                index_min_qu = qu_len.index(min(qu_len))
                if self.reception.Queue_to_room[0].has_corona:
                    self.Rooms[index_min_qu].corona_patients_queue.append(self.reception.Queue_to_room.pop(0))
                else:
                    self.Rooms[index_min_qu].normal_patients_queue.append(self.reception.Queue_to_room.pop(0))

            len_queue_corona_rooms = 0
            len_queue_normal_rooms = 0
            number_of_normal_pat_in_rooms = 0
            number_of_corona_pat_in_rooms = 0

            time3 = time.time()
            self.time_calculate_min_len += time3 - time2
            
            # check bored time in rooms queue
            for i in range(M):
                time3 = time.time()

                for patient in self.Rooms[i].corona_patients_queue:
                    if clock - patient.arrival_time - patient.service_time > patient.bored_time:
                        left_rooms_corona_pats += 1
                        self.Rooms[i].corona_patients_queue.remove(patient)
                        self.number_of_patients -= 1
                for patient in self.Rooms[i].normal_patients_queue:
                    if clock - patient.arrival_time - patient.service_time > patient.bored_time:
                        left_rooms_normal_pats += 1
                        self.Rooms[i].normal_patients_queue.remove(patient)
                        self.number_of_patients -= 1

                time4 = time.time()
                self.time_tired_room_took += time4 - time3
                
                # check up each room :

                temp_number_of_pat_finished_check_up = self.Rooms[i].check_up(clock)

                time5 = time.time()
                self.time_room_took += time5 - time4

                self.number_of_patients -= temp_number_of_pat_finished_check_up

                len_queue_corona_rooms += len(self.Rooms[i].corona_patients_queue)
                len_queue_normal_rooms += len(self.Rooms[i].normal_patients_queue)
                number_of_corona_pat_in_rooms = self.Rooms[i].number_of_corona_pat_in_room
                number_of_normal_pat_in_rooms = self.Rooms[i].number_of_normal_pat_in_room
                len_rooms_queue[i] += len_queue_normal_rooms + len_queue_corona_rooms

                self.time_calculation_in1 += time.time() - time5

            time5 = time.time()

            total_len_reception_queue+=(len_reception_queue_corona+len_reception_queue_normal)
            dict_people_in_queue_corona[clock] = len_queue_corona_rooms+len_reception_queue_corona
            dict_people_in_queue_normal[clock] = len_queue_normal_rooms+len_reception_queue_normal
            dict_people_in_queue[clock] = dict_people_in_queue_normal[clock] + dict_people_in_queue_corona[clock]

            dict_people_in_system_corona[clock] = dict_people_in_queue_corona[clock] + number_of_corona_pat_in_rooms + (
                    self.reception.reception_busy and self.reception.patient_in_reception.has_corona)
            dict_people_in_system_normal[clock] = dict_people_in_queue_normal[clock] + number_of_normal_pat_in_rooms + (
                    self.reception.reception_busy and not self.reception.patient_in_reception.has_corona)

            dict_people_in_system[clock] = dict_people_in_system_corona[clock] + dict_people_in_system_normal[clock]

            self.time_calculation_in2 += time.time() - time5

            ###################################################################3
            if self.number_of_patients == 0:
                temp_counter += 1
            if temp_counter == 2:
                break
            clock += 1

        ###################################### Outputs:

        info = {}
        normal_totals = n - self.corona_totals
        left_pats = left_reception_normal_pats + left_reception_corona_pats + left_rooms_corona_pats + left_rooms_normal_pats

        total_wait_in_queue_corona = self.reception.total_time_wait_in_q_corona_pats + sum(
            [self.Rooms[i].total_time_wait_in_room_q_corona for i in range(M)])
        total_wait_in_queue_normal = self.reception.total_time_wait_in_q_normal_pats + sum(
            [self.Rooms[i].total_time_wait_in_room_q_normal for i in range(M)])

        info["mean_time_wait_in_queue"] = (total_wait_in_queue_normal + total_wait_in_queue_corona) / (
                len(self.persons_normal) + len(perosons_corona) - left_pats)
        info["mean_time_wait_in_queue_corona"] = (total_wait_in_queue_corona) / (
                self.corona_totals - (left_rooms_corona_pats + left_reception_corona_pats))
        info["mean_time_wait_in_queue_normal"] = (total_wait_in_queue_normal) / (
                normal_totals - (left_rooms_normal_pats + left_reception_normal_pats))
        total_time_in_system_corona_pats = sum([self.Rooms[i].total_time_in_sys_corona_pats for i in range(M)])
        total_time_in_system_normal_pats = sum([self.Rooms[i].total_time_in_sys_normal_pats for i in range(M)])

        info["mean_time_in_system_for_all"] = (total_time_in_system_corona_pats + total_time_in_system_normal_pats) / (
                len(self.persons_normal) + len(perosons_corona) - left_pats)
        info["mean_time_in_system_for_corona_pat"] = total_time_in_system_corona_pats / (
                self.corona_totals - (left_rooms_corona_pats + left_reception_corona_pats))
        info["mean_time_in_system_for_normal_pat"] = total_time_in_system_normal_pats / (
                normal_totals - (left_rooms_normal_pats + left_reception_normal_pats))

        ##### mean left pats :
        info["mean_left_pat"] = left_pats / (len(self.persons_normal) + len(perosons_corona)) * 100

        ##### mean reception queue and each room  :


        info["mean_len_reception_queue"] = total_len_reception_queue/clock
        for i in range(M):
            info["mean_len_room_queue_" + str(i)] = len_rooms_queue[i] / clock

        info["people_in_queue_in_each_clk"] = dict_people_in_queue
        info["corona_people_in_queue_in_each_clk"] = dict_people_in_queue_corona
        info["normal_people_in_queue_in_each_clk"] = dict_people_in_queue_normal

        info["people_in_system_in_each_clk"] = dict_people_in_system
        info["corona_people_in_system_in_each_clk"] = dict_people_in_system_corona
        info["normal_people_in_system_in_each_clk"] = dict_people_in_system_normal

        return info


#######################################################################################################################################

n = 10_000_000  # number of patinets


M = 2  # number of rooms
alpha = 30  # mean time to bored
u = 8  # service rate for paziresh
landa = 5  # arrival rate

number_of_doctors_per_room = [2, 3]  # , 2, 3 ]
mean_check_up_time = [[10, 12], [6, 8,13]]

start_time = time.time()

perosons = []
perosons_corona = []
perosons_normal = []

service_time_for_paziresh = [i + 1 for i in list(numpy.random.poisson(u, n))]
arrival_time = list(numpy.random.poisson(landa, n))

for i in range(len(arrival_time)):
    if i == 0:
        continue
    arrival_time[i] = arrival_time[i - 1] + arrival_time[i]
corona_totals = int(0.1 * n)

type_1_list_idx = random.sample(range(n), corona_totals)
type_1_list_idx.sort()
corona_temp_index = 0
print("now time is :",time.time())
for i in range(n):
    if i == type_1_list_idx[corona_temp_index]:
        perosons_corona.append(PERSON(True, alpha, arrival_time[i], service_time_for_paziresh[i]))
        if corona_temp_index < corona_totals - 1:
            corona_temp_index += 1
    else:
        perosons_normal.append(PERSON(False, alpha, arrival_time[i], service_time_for_paziresh[i]))

print("time after persons is :",time.time())

start_time2 = time.time()

Hospital = hospital(M, number_of_doctors_per_room, mean_check_up_time, perosons_corona, perosons_normal, corona_totals)

info = Hospital.start_simulation()
print("after start simulation time is :",time.time())
print("\nMy program took", time.time() - start_time2, "to run\n")

####################################################################################### 1-4

print("mean_time_in_system_for_all: ", info["mean_time_in_system_for_all"])
print("mean_time_in_system_for_corona_pat", info["mean_time_in_system_for_corona_pat"])
print("mean_time_in_system_for_normal_pat", info["mean_time_in_system_for_normal_pat"])
print("\n#######################################\n")
print("mean time wait in queue: ", info["mean_time_wait_in_queue"])
print("mean time in queue corona pats: ", info["mean_time_wait_in_queue_corona"])
print("mean time in queue normal pats: ", info["mean_time_wait_in_queue_normal"])
print("\n#######################################\n")
print("left system: ", info["mean_left_pat"], " percent")
print("mean len of reception queue : ", info["mean_len_reception_queue"])
for i in range(M):
    print("mean_len_room_queue_" + str(i), info["mean_len_room_queue_" + str(i)])

#####################################################################################5
sigma=0
normal_response_times=[]
normal_waits=[]
for pat in Hospital.persons_normal:
    if pat.respons_time!=None:
        normal_response_times.append(pat.respons_time)
    if pat.total_wait!=None:
        normal_waits.append(pat.total_wait)
        sigma+=((pat.total_wait-info["mean_time_wait_in_queue"])*(pat.total_wait-info["mean_time_wait_in_queue"]))

corona_rexpons_times=[]
corona_waits=[]

for pat in Hospital.persons_corona:
    if pat.respons_time!=None:
        corona_rexpons_times.append(pat.respons_time)
    if pat.total_wait!=None:
        corona_waits.append(pat.total_wait)
        sigma+=((pat.total_wait-info["mean_time_wait_in_queue"])*(pat.total_wait-info["mean_time_wait_in_queue"]))

print("n",n-info["mean_left_pat"]*n/100-1)
sigma=sigma/(n-info["mean_left_pat"]*n/100-1)
print("sigma is ",sigma)

sigma=math.sqrt(sigma)
num_customer=((1.96*sigma)/(0.01*info["mean_time_wait_in_queue"]))**2
print("################################################################\n")
print("number of patient to reach  0.95 accurany is: ",num_customer)
#####################################################################################6
"""
mean_len_q=1
while mean_len_q>0.01:
    Hospital = hospital(M, number_of_doctors_per_room, mean_check_up_time, perosons_corona,perosons_normal, corona_totals)

    info = Hospital.start_simulation()

    mean_len_q= sum([info["mean_len_room_queue_" + str(i)] for i in range(M)])/M

    print("check up time ", mean_check_up_time)
    print("\nmeean len qu room ", mean_len_q)
    for k in range(len(mean_check_up_time)):
        for i in range(len(mean_check_up_time[k])):
            if mean_check_up_time[k][i]:
                mean_check_up_time[k][i]-=1



print("\nmean check up time when len of rooms queue is 0 : ",mean_check_up_time)
"""
#############################################################################1 emtiazi





plt.hist(normal_response_times,30, alpha=0.5, label='normal pats',color='g')
plt.hist(corona_rexpons_times, 30, alpha=0.5, label='corona pats',color='r')

plt.legend(loc='upper right')
plt.title("Response Time Frequency")
plt.xlabel("Response time")
plt.ylabel("Frequency")
plt.show()

################################################################################2 emtizi
plt.hist(normal_waits, 30, alpha=0.5, label='normal pats',color='g')
plt.hist(corona_waits, 30, alpha=0.5, label='corona pats',color='r')

plt.legend(loc='upper right')
plt.title("Waiting Time Frequency")
plt.xlabel("WAinting time")
plt.ylabel("Frequency")
plt.show()
################################################################################3 emtiazi
plt.title("people in sys frequency")
plt.xlabel("clock*100")
plt.ylabel("number of pats in system")
plt.plot([i//100 for i in info["normal_people_in_system_in_each_clk"].keys()],info["normal_people_in_system_in_each_clk"].values(),color='green',label='corona pats')
plt.plot([i//100 for i in info["corona_people_in_system_in_each_clk"].keys()],info["corona_people_in_system_in_each_clk"].values(), color='red',label="normal pats")
#plt.legend()
plt.show()
###############################################################################4 emtiazi
plt.title("people_in_sys_in_each_clk")
plt.xlabel("clock")
plt.ylabel("number of pats in system")
plt.plot(info["people_in_system_in_each_clk"].keys(),info["people_in_system_in_each_clk"].values(), color='blue',label='all')
plt.plot(info["normal_people_in_system_in_each_clk"].keys(),info["normal_people_in_system_in_each_clk"].values(), color='green',label='corona pats')
plt.plot(info["corona_people_in_system_in_each_clk"].keys(),info["corona_people_in_system_in_each_clk"].values(), color='red',label="normal pats")
#plt.legend()
plt.show()
################################################################################5 emtiazi
plt.title("people_in_queue_in_each_clk")
plt.xlabel("clock")
plt.ylabel("number of pats in q")
plt.plot(info["people_in_queue_in_each_clk"].keys(),info["people_in_queue_in_each_clk"].values(), color='blue',label='all')
plt.plot(info["normal_people_in_queue_in_each_clk"].keys(),info["normal_people_in_queue_in_each_clk"].values(), color='green',label='corona pats')
plt.plot(info["corona_people_in_queue_in_each_clk"].keys(),info["corona_people_in_queue_in_each_clk"].values(), color='red',label="normal pats")
#plt.legend()
plt.show()

print("end   : ", time.time()-start_time2)
