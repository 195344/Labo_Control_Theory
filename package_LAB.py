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
    The function "FO_RT" needs to be included in a "for or while loop".
    
    :MV: input vector
    :Kp: process gain
    :T: lag time constant [s]
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
    
    #Add FF
    
    if len(PV) == 0:
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
            MVD.append(0)
        
        
    #Integral part 
    if Ti > 0:
        if len(MVI) !=0:
            MVI.append(Kc * (Ts / Ti) * E[-1] + MVI[-1])
        else:
            MVI.append(0)
        
        
    #Proportional part
    MVP.append(Kc * E[-1])
              
    
    #MV limit
    MV_next = MVP[-1] + MVI[-1] + MVD[-1]
    if MV_next >= MVMax:
        MV_next = MVMax
    if MV_next <= MVMin:
        MV_next = MVMin
                   
    MV.append(MV_next)