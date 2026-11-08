import numpy as np

def experimenteer_met_xi_formule():
    print("==========================================================")
    print("ETRT REKENKAMER: ZUIVERE XI-UNIFICATIE FORMULE")
    print("Doel: Balans vinden waarbij Neutron (UDD) iets zwaarder is dan Proton (UUD)")
    print("==========================================================\n")
    
    y_vac = 10.0   # Onverstoorde roosterafstand in het vacuüm
    xi = 0.5       # De universele halfwaarde-radius (TR=0.50 op r=xi)
    
    # Micro-kalibratie rond de ontdekte sweet spot
    y_up_values = np.arange(5.3, 5.6, 0.05)       
    y_down_values = np.arange(14.8, 15.3, 0.05)
    
    print(f"Vaste parameters: y_vac = {y_vac}, Universele xi = {xi}\n")
    print(f"{'y_UP':<6} | {'y_DOWN':<6} | {'Massa Up':<10} | {'Massa Down':<10} | {'Proton (UUD)':<14} | {'Neutron (UDD)':<14} | {'Ratio (N/P)'}")
    print("-" * 90)
    
    for y_up in y_up_values:
        for y_down in y_down_values:
            
            # --- 1. BEREKENING UP-QUARK ---
            # Macroscopische dichtheid * Microscopische deuk op r = y/2
            tr_up = ((y_vac / y_up)**3) * (((y_up/2) / ((y_up/2) + xi))**2)
            delta_tr_up = tr_up - 1.0
            vol_up = y_up**3
            m_up = vol_up * (delta_tr_up**2)
            
            # --- 2. BEREKENING DOWN-QUARK ---
            tr_down = ((y_vac / y_down)**3) * (((y_down/2) / ((y_down/2) + xi))**2)
            delta_tr_down = tr_down - 1.0
            vol_down = y_down**3
            m_down = vol_down * (delta_tr_down**2)
            
            # --- 3. NUCLEONEN SAMENSTELLEN ---
            m_proton = 2*m_up + m_down      # Proton = 2x Compressie + 1x Expansie
            m_neutron = m_up + 2*m_down     # Neutron = 1x Compressie + 2x Expansie
            
            ratio = m_neutron / m_proton
            
            print(f"{y_up:<6.1f} | {y_down:<6.1f} | {m_up:<10.0f} | {m_down:<10.0f} | {m_proton:<14.0f} | {m_neutron:<14.0f} | {ratio:.4f}")

    print("-" * 90)
    print("\nAls de ratio de ~1.001 passeert, hebben we de fysieke waarden voor y_up en y_down gevonden!")

if __name__ == "__main__":
    experimenteer_met_xi_formule()