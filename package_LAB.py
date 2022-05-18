import numpy as np

import matplotlib.pyplot as plt
from IPython.display import display, clear_output

#-----------------------------------
def LeadLag_RT(MV,Kp,Tlead,Tlag,Ts,PV,PVInit=0,method='EBD'):
    
    """
    The function "LeadLag_RT" needs to be included in a "for or while loop".
    
    :MV: input vector
    :Kp: process gain
    :Tlead: lead time constant [s]
    :Tlag: lag time constant [s]
    :Ts: sampling period [s]
    :PV: output vector
    :PVInit: (optional: default value is 0)
    :method: discretisation method (optional: default value is 'EBD')
        EBD: Euler Backward difference
        EFD: Euler Forward difference
        TRAP: Trapezoïdal method
    
    The function "FO_RT" appends a value to the output vector "PV".
    The appended value is obtained from a recurrent equation that depends on the discretisation method.
    """    
    
    if (Tlag != 0):
        K = Ts/Tlag
        if len(PV) == 0:
            PV.append(PVInit)
        else: # MV[k+1] is MV[-1] and MV[k] is MV[-2]
            if method == 'EBD':
                PV.append ((1/(1+K))*PV[-1] + ((K*Kp)/(1+K))*((1+(Tlead/Ts))*MV[-1]-(Tlead/Ts)*MV[-2]))
            elif method == 'EFD':
                PV.append ((1-K)*PV[-1] + K*Kp*((Tlead/Ts)*MV[-1]+(1-(Tlead/Ts))*MV[-2]))
            elif method == 'TRAP':
                PV.append((1/(2*Tlag+Ts))*((2*Tlag-Ts)*PV[-1]+(2*Tlead+Ts)*Kp*MV[-1]+(Ts-2*Tlead)*Kp*MV[-2]))
            else:
                PV.append((1/(1+K))*PV[-1] + (K*Kp/(1+K))*((1+(Tlead/Ts))*MV[-1]-(Tlead/Ts)*MV[-2]))
    else:
        PV.append(Kp*MV[-1])
        
        
        
        
#-----------------------------------
def PID_RT(SP, PV, Man, MVMan, MVFF, Kc, Ti, Td, alpha, Ts, MVMin, MVMax, MV, MVP, MVI, MVD, E, ManFF, PVInit, method='EBD-EBD'):
    
    """
    
    :SP: SP (SetPoint) vector
    :PV: PV (Process value) vector
    :Man: Man (Manual controller mode) vector (true or false)
    :MVMan: MVMan (Manual value for MV) vector
    :MVFF: MVFF (Feedforward) vector
    
    :Kc: controller gain
    :Ti: integral time constant [s]
    :Td: derivative period [s]
    :alpha: Tfd = alpha*Td, Tfd is the derivative filter time constant [s]
    :Ts: sampling period [s] 
    
    :MVMin: minimum value of MV (used for saturation and anti wind-up)
    :MVMax: maximum value of MV (used for saturation and anti wind-up)
    
    :MV: MV (Manipulated Value) vector
    :MVP: MVP (Proportional part of MV) vector
    :MVI: MVI (Integral part of MV) vector
    :MVD: MVD (Derivative part of MV) vector
    :E: E (control error) vector
    
    :ManFF Activated FF in manual mode (optional: default boolean value is false)
    :PVInit: Initial value for PV (optional: default value is 0): used if PID_RT in ran first in the first sequence and no value of PV is available yet. 
    
    The function "PID_RT" appends new values to the vectors "MV", "MVP", "MVI", and "MVD". The appended values are based on the PID algorithm, the controller mode, and the feedforward. Note that the saturation of "MV" within the limits [MVMin MVMax] is implemented with an anti wind-up.
    """    
    
    
    
    if len(PV) == 0:
        E.append(SP[-1] - PVInit)
    else:
        E.append(SP[-1] - PV[-1])
        
        
    #Derivative part
    Tfd = alpha * Td
    if len(MVD) !=0:
        if len(E) == 1:
            MVD.append((Tfd / (Tfd + Ts)) * MVD[-1] + ((Kc * Td) / (Tfd + Ts)) * (E[-1]))
        else:
            MVD.append((Tfd / (Tfd + Ts)) * MVD[-1] + ((Kc * Td) / (Tfd + Ts)) * (E[-1] - E[-2]))
    else:
        if len(E) == 1:
            MVD.append((Kc * Td) / (Tfd + Ts) * (E[-1]))
        else:
            MVD.append((Kc * Td) / (Tfd + Ts) * (E[-1] - E[-2]))
        
        
    #Integral part 
    if len(MVI) == 0:
        MVI.append((Kc * Ts / Ti) * E[-1])
    else:
        MVI.append(((Kc * Ts / Ti) * E[-1]) + MVI[-1])
        
        
    #Proportional part
    MVP.append(Kc * E[-1])
        
        
        
    if len(MVFF) == 0:
        MVFFint = 0
    else:
        MVFFint = MVFF[-1]
        
    if bool(Man[-1]) == True:
        if ManFF:
            MVI[-1] = MVMan[-1] - MVP[-1] - MVD[-1]
        else:
            MVI[-1] = MVMan[-1] - MVP[-1] - MVD[-1] - MVFFint
              
    
    #MV limit
    MV_next = MVP[-1] + MVI[-1] + MVD[-1] + MVFFint
    
    if MV_next >= MVMax:
        MVI[-1] = MVMax - MVP[-1] - MVD[-1] - MVFFint
        MV_next = MVMax
        
    if MV_next <= MVMin:
        MVI[-1] = MVMin - MVP[-1] - MVD[-1] - MVFFint
        MV_next = MVMin
                   
    MV.append(MV_next)
    
    
    
    
#-----------------------------------
def IMCTuning(K, Tlag1, Tlag2=0, theta=0, gamma=0.5, process='FOPDT-PI'):
    
    """
    
    :K: process gain
    :Tlag1: first (or main) lag time constant [s]
    :Tlag2: second lag time constant [s] (optional: default value is 0.0)
    :theta: delay [s] (optional: default value is 0.0)
    :gamma: used to compute the desired closed-loop time constant TCLP [s] (range for gamma [0.2 ... 0.9], optional: default value is 0.5)
    :process: (optional: default value is 'FOPDT-PI')
        FOPDT-PI: First Order Plus Dead Time for PI control (IMC tuning: case G)
        FOPDT-PID: First Order Plus Dead Time for PID control (IMC tuning: case H)
        SOPDT: Second Order Plus Dead Time for PID control (IMC tuning: case I)
     
    :return: PID  controller Kc, Ti and Td 
    """
    
    TCLP = gamma * Tlag1 
    
    if process == 'FOPDT-PI':
        
        Kc = (Tlag1 / (TCLP + theta)) / K
        Ti = Tlag1
        return Kc, Ti, 0
        
    if process == 'FOPDT-PID':
        
        Kc = ((Tlag1 + (theta / 2)) / (TCLP + (theta / 2))) / K
        Ti = Tlag1 + (theta / 2)
        Td = (Tlag1 * theta) / ((2 * Tlag1) + theta)
        
        return Kc, Ti, Td
        
    if process == 'SOPDT':
        
        Kc = ((Tlag1 + Tlag2) / (TCLP + theta)) / K
        Ti = Tlag1 + Tlag2
        Td = (Tlag1 * Tlag2) / (Tlag1 + Tlag2)
        return Kc, Ti, Td
    
    return 0, 0, 0




#-----------------------------------        
class PID:
    
    def __init__(self, parameters):
        
        self.parameters = parameters
        self.parameters['Ti'] = parameters['Ti'] if 'Ti' in parameters else 10.0
        self.parameters['Td'] = parameters['Td'] if 'Td' in parameters else 0.0
        self.parameters['Kc'] = parameters['Kc'] if 'Kc' in parameters else 0.0
        self.parameters['alpha'] = parameters['alpha'] if 'alpha' in parameters else 0.0


#-----------------------------------
def Margins(P, C, omega, Show = True):
    
    """
    :P: Process as defined by the class "Process"
        Use the following command to define the default process wich is simply a unit gain process:
            P = Process({})
        
        A delay, two lead time constants and 2 lag constants can be added.
    
        Use the following commands for a SOPDT process:
            P.parameters['Kp'] = 1.1
            P.parameters['Tlag1'] = 0.0
            P.parameters['Tlag2'] = 2.0
            P.parameters['theta'] = 2.0
            
        Use the following commands for a unit gain lead-lag process:
            P.parameters['Tlag1'] = 10.0
            P.parameters['Tlead1'] = 15.0
            
    :C: PID controller as defined by the class "PID"
        Use the following command to define the default PID controller wich is simply a unit gain PI controller with Ti = 10s:
            C = PID({})
            
        Use the following commands for a PID controller:
            C.parameters['Kc'] = 2.0
            c.parameters['Ti'] = 100.0
            C.parameters['Td'] = 10.0
            C.parameters['alpha'] = 1.0
    
    :omega: frequency vector (rad/s); generated by a commad of type "omega = np.logspace(-2, 2, 10000)"
    
    :return: omegau, Gm (Gain margin in dB), omegac, Pm (Phase margin in °)
    
    The function "Margins" generates the Bode diagram of the loop gain L = P*C with the stability margins/
    """

    s = 1j*omega
    Ptheta = np.exp(-P.parameters['theta']*s)
    PGain = P.parameters['Kp']*np.ones_like(Ptheta)
    PLag1 = 1/(P.parameters['Tlag1']*s + 1)
    PLag2 = 1/(P.parameters['Tlag2']*s + 1)
    PLead1 = P.parameters['Tlead1']*s + 1
    PLead2 = P.parameters['Tlead2']*s + 1

    Ps = np.multiply(Ptheta,PGain)
    Ps = np.multiply(Ps,PLag1)
    Ps = np.multiply(Ps,PLag2)
    Ps = np.multiply(Ps,PLead1)
    Ps = np.multiply(Ps,PLead2)

    Cs = C.parameters['Kc']*(1 + (1/C.parameters['Ti'])*s + (C.parameters['Td']*s)/(C.parameters['alpha'] * C.parameters['Td'] * s + 1))
    Ls = np.multiply(Ps,Cs)

    if Show == True:
        fig, (ax_gain, ax_phase) = plt.subplots(2,1)
        fig.set_figheight(12)
        fig.set_figwidth(22)
    
        gain = 20*np.log10(np.abs(Ls))
        phase = (180/np.pi)*np.unwrap(np.angle(Ls))
    
        # Gain proche de 0
        gain0 = np.argmin(np.abs(gain))

        # Phase proche de -180
        phase180 = np.argmin(np.abs(phase - (-180)))
    
        # Trouver les valeurs de Gain et phase de marge
        gainCross = omega[gain0]
        phaseCross = omega[phase180]
    
        ax_gain.set_title('Diagramme de Bode,Marge de Phase :  {}°, Marge de gain : {} dB'.format(np.around(180+phase[gain0], decimals=3), np.around(-gain[phase180], decimals=3)))
    
    #Gain
    
        point1 = [phaseCross, gain[phase180]]
        point2 = [phaseCross, 0]
        x_values = [point1[0], point2[0]]
        y_values = [point1[1], point2[1]]
        ax_gain.plot(x_values, y_values, label='Gain margin')

        ax_gain.semilogx(omega,gain,label='L(s)')
        ax_gain.axhline(y=0, color='r', linestyle='-', label='0 dB')
        gain_min = np.min(20*np.log10(np.abs(Ls)/5))
        gain_max = np.max(20*np.log10(np.abs(Ls)*5))
        ax_gain.set_xlim([np.min(omega), np.max(omega)])
        ax_gain.set_ylim([gain_min, gain_max])
        ax_gain.set_ylabel('Amplitude |L| [db]')
        ax_gain.legend(loc='best')
  
    # Phase


        point1 = [gainCross, -180]
        point2 = [gainCross, phase[gain0]]
        x_values = [point1[0], point2[0]]
        y_values = [point1[1], point2[1]]
        ax_phase.plot(x_values, y_values, label='Phase margin')

        ax_phase.semilogx(omega, phase, label='L(s)') 
        ax_phase.axhline(y=-180, color='r', linestyle='-', label='-180 deg')
        ax_phase.set_xlim([np.min(omega), np.max(omega)])
        ph_min = np.min((180/np.pi)*np.unwrap(np.angle(Ls))) - 10
        ph_max = np.max((180/np.pi)*np.unwrap(np.angle(Ls))) + 10
        ax_phase.set_ylim([np.max([ph_min, -200]), ph_max])
        ax_phase.set_ylabel(r'Phase $\angle L$ [°]')
        ax_phase.legend(loc='best')
    else:
        return Ls
