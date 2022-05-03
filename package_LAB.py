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
def PID_RT(SP, PV, Man, MVMan, MVFF, Kc, Ti, Td, alpha, Ts, MVMin, MVMax, MV, MVP, MVI, MVD, E, ManFF=False, PVInit=0, method='EBD-EBD'):
    
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
    
    
    
    if len(PV) or len(E) == 0:
        E.append(SP[-1] - PVInit)
    else:
        E.append(SP[-1] - PV[-1])
        
        
    #Derivative part
    Tfd = alpha * Td
    if Td > 0:
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
    if Ti > 0:
        if len(MVI) !=0:
            MVI.append(Kc * (Ts / Ti) * E[-1] + MVI[-1])
        else:
            MVI.append(Kc * (Ts / Ti) * E[-1])
        
        
    #Proportional part
    MVP.append(Kc * E[-1])
    
    
    #Manual mode + anti wind-up
    if ManFF:
        MVFFint = MVFF[-1]
    else:
        MVFFint = 0
        
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
    