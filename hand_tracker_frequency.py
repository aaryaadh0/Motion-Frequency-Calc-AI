import cv2
import mediapipe as mp
import time
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# --- MLP SETUP ---
W1 = np.array([[0.5, -0.3, 0.8, 0.2],    # input1 to hidden
               [0.6, 0.1, -0.5, 0.7]])   # input2 to hidden
b1 = np.array([0.1, 0.2, 0.0, -0.1])     # bias for hidden layer

W2 = np.array([0.4, -0.6, 0.3, 0.2])     # hidden to output
b2 = 0.05                                # bias for output

def relu(x):
    return np.maximum(0, x)
#takes two numerical inputs feeds them into the MLP and calculate
#output by first computing the hidden layer activation using W1 and B! and applies relu function 
#and finally computes the output using W2 and B2
def mlp_predict_numeric(freq1, freq2):
    inputs = np.array([freq1, freq2])
    hidden = relu(np.dot(inputs, W1) + b1)
    output = np.dot(hidden, W2) + b2
    return output
#calculates the MSE loss betwn predicted value and the actual target by squaring the difference
def mse_loss(pred, target):
    return (pred - target) ** 2

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --- Webcam ---
cap = cv2.VideoCapture(0)

# --- Buffers ---
y_values_hands = [deque(maxlen=50), deque(maxlen=50)] #Yo le dui ota sliding window banaucha 
#(deque ko help le), jasma hami dui ota haat ko last 50 y-coordinate (vertical position) values store garna sakchhau.
timestamps_peaks_hands = [[], []]# stores the timestamps of peak position
last_peak_time_hands = [0, 0]
window_seconds = 3

# --- Frequency buffers for plotting ---
freq_values = [deque(maxlen=100), deque(maxlen=100)]
time_values = deque(maxlen=100)

start_time = time.time()

def is_peak(vals):
    if len(vals) < 3:
        return False
    return vals[-2] < vals[-3] and vals[-2] < vals[-1]
#Yo function le last 3 ota value haru ma middle waala value (vals[-2]) ko check garcha 
# ki tyo value usko agadi ra pachi ko value bhanda sano cha ki chaina.
#If sano cha bhane yo local minimum ho, jasko matlab ho valley 
# (haat ko movement ma haat mathi-tala ko sabai bhanda tala ko point).

def update(frame):
    global last_peak_time_hands, timestamps_peaks_hands

    success, img = cap.read()
    if not success:
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    current_time = time.time() - start_time

    frequencies = [0, 0]

    if results.multi_hand_landmarks:
        for i, handLms in enumerate(results.multi_hand_landmarks):
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            wrist_y = handLms.landmark[0].y
            y_values_hands[i].append(wrist_y)

            if is_peak(list(y_values_hands[i])[-3:]):
                if current_time - last_peak_time_hands[i] > 0.3:
                    timestamps_peaks_hands[i].append(current_time)
                    last_peak_time_hands[i] = current_time

            timestamps_peaks_hands[i] = [t for t in timestamps_peaks_hands[i] if current_time - t <= window_seconds]

            frequencies[i] = len(timestamps_peaks_hands[i]) / window_seconds if timestamps_peaks_hands[i] else 0

    # Show frequency on webcam
    for i, freq in enumerate(frequencies):
        cv2.putText(img, f'Hand {i+1} Freq: {freq:.2f} Hz', (30, 50 + i*40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Webcam Feed !!!", img)

    # MLP numeric prediction
    score = mlp_predict_numeric(frequencies[0], frequencies[1])
    target = 1.0  # temp expected value ho, jasma model ko prediction score compare garinxa
    
    loss = mse_loss(score, target)#model ko prediction ra target bich ko diff. 
    #ko square calculate garxa
    
    #example:
    #score = 1.375 (predicted by your MLP)
    #target = 1.0 (expected value)
    #loss=(1.375−1.0)^2

    print(f"MLP output: {score:.3f}, Loss: {loss:.3f}")

    # Plotting
    time_values.append(current_time)
    for i in range(2):
        freq_values[i].append(frequencies[i])

    ax.clear()
    ax.plot(list(time_values), list(freq_values[0]), 'r-', label='Hand 1 Frequency (Hz)')
    ax.plot(list(time_values), list(freq_values[1]), 'b-', label='Hand 2 Frequency (Hz)')
    ax.set_xlim(max(0, current_time - 10), current_time)
    max_y = max(max(freq_values[0], default=1), max(freq_values[1], default=1)) + 0.5
    ax.set_ylim(0, max_y)
    ax.set_title('Live Hand Movement Frequency')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.legend(loc='upper right')

    return line1, line2

fig, ax = plt.subplots()
line1, = ax.plot([], [], 'r-', label='Hand 1 Frequency')
line2, = ax.plot([], [], 'b-', label='Hand 2 Frequency')

ani = FuncAnimation(fig, update, interval=50)
plt.show()

cap.release()
cv2.destroyAllWindows()



#frequency calculate 
#Frequency (Hz)= Number of Peaks/Window Duration (seconds)

