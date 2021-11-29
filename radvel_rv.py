import numpy as np

def rv_drive(t, orbel):

    """RV Drive

    Args:

        t (array of floats): times of observations

        orbel (array of floats): [per, tp, e, om, K].\

            Omega is expected to be\

            in radians

        use_c_kepler_solver (bool): (default: True) If \

            True use the Kepler solver written in C, else \

            use the Python/NumPy version.

    Returns:

        rv: (array of floats): radial velocity model

    """

    # unpack array of parameters

    per, tp, e, om, k = orbel

    # Performance boost for circular orbits

    if e == 0.0:

        m = 2 * np.pi * (((t - tp) / per) - np.floor((t - tp) / per))

        return k * np.cos(m + om)

    if per < 0:

        per = 1e-4

    if e < 0:

        e = 0

    if e > 0.99:
        e = 0.99



    # Calculate the approximate eccentric anomaly, E1, via the mean anomaly  M.
    nu = true_anomaly(t, tp, per, e)
    rv = k * (np.cos(nu + om) + e * np.cos(om))


    return rv


def timetrans_to_timeperi(tc, per, ecc, omega):
    """
    Convert Time of Transit to Time of Periastron Passage

​

    Args:

        tc (float): time of transit

        per (float): period [days]

        ecc (float): eccentricity

        omega (float): longitude of periastron (radians)

​

    Returns:

        float: time of periastron passage

​

    """

    try:

        if ecc >= 1:

            return tc

    except ValueError:

        pass


    f = np.pi/2 - omega

    ee = 2 * np.arctan(np.tan(f/2) * np.sqrt((1-ecc)/(1+ecc)))  # eccentric anomaly

    tp = tc - per/(2*np.pi) * (ee - ecc*np.sin(ee))      # time of periastron



    return tp


def timeperi_to_timetrans(tp, per, ecc, omega, secondary=False):

    """

    Convert Time of Periastron to Time of Transit

​

    Args:

        tp (float): time of periastron

        per (float): period [days]

        ecc (float): eccentricity

        omega (float): argument of peri (radians)

        secondary (bool): calculate time of secondary eclipse instead

​

    Returns:

        float: time of inferior conjunction (time of transit if system is transiting)

​

    """

    try:

        if ecc >= 1:

            return tp

    except ValueError:

        pass


    if secondary:

        f = 3*np.pi/2 - omega                                       # true anomaly during secondary eclipse

        ee = 2 * np.arctan(np.tan(f/2) * np.sqrt((1-ecc)/(1+ecc)))  # eccentric anomaly


        # ensure that ee is between 0 and 2*pi (always the eclipse AFTER tp)

        if isinstance(ee, np.float64):

            ee = ee + 2 * np.pi

        else:

            ee[ee < 0.0] = ee + 2 * np.pi

    else:

        f = np.pi/2 - omega                                         # true anomaly during transit

        ee = 2 * np.arctan(np.tan(f/2) * np.sqrt((1-ecc)/(1+ecc)))  # eccentric anomaly


    tc = tp + per/(2*np.pi) * (ee - ecc*np.sin(ee))         # time of conjunction

    return tc

def true_anomaly(t, tp, per, e):

    """

    Calculate the true anomaly for a given time, period, eccentricity.

​

    Args:

        t (array): array of times in JD

        tp (float): time of periastron, same units as t

        per (float): orbital period in days

        e (float): eccentricity

​

    Returns:

        array: true anomoly at each time

    """


    # f in Murray and Dermott p. 27

    m = 2 * np.pi * (((t - tp) / per) - np.floor((t - tp) / per))

    eccarr = np.zeros(t.size) + e

    e1 = kepler(m, eccarr)

    n1 = 1.0 + e

    n2 = 1.0 - e

    nu = 2.0 * np.arctan((n1 / n2)**0.5 * np.tan(e1 / 2.0))


    return nu

def kepler(Marr, eccarr):
    """Solve Kepler's Equation
    Args:
        Marr (array): input Mean anomaly
        eccarr (array): eccentricity
    Returns:
        array: eccentric anomaly
    """

    conv = 1.0e-12  # convergence criterion
    k = 0.85

    Earr = Marr + np.sign(np.sin(Marr)) * k * eccarr  # first guess at E
    # fiarr should go to zero when converges
    fiarr = ( Earr - eccarr * np.sin(Earr) - Marr)
    convd = np.where(np.abs(fiarr) > conv)[0]  # which indices have not converged
    nd = len(convd)  # number of unconverged elements
    count = 0

    while nd > 0:  # while unconverged elements exist
        count += 1

        M = Marr[convd]  # just the unconverged elements ...
        ecc = eccarr[convd]
        E = Earr[convd]

        fi = fiarr[convd]  # fi = E - e*np.sin(E)-M    ; should go to 0
        fip = 1 - ecc * np.cos(E)  # d/dE(fi) ;i.e.,  fi^(prime)
        fipp = ecc * np.sin(E)  # d/dE(d/dE(fi)) ;i.e.,  fi^(\prime\prime)
        fippp = 1 - fip  # d/dE(d/dE(d/dE(fi))) ;i.e.,  fi^(\prime\prime\prime)

        # first, second, and third order corrections to E
        d1 = -fi / fip
        d2 = -fi / (fip + d1 * fipp / 2.0)
        d3 = -fi / (fip + d2 * fipp / 2.0 + d2 * d2 * fippp / 6.0)
        E = E + d3
        Earr[convd] = E
        fiarr = ( Earr - eccarr * np.sin( Earr ) - Marr) # how well did we do?
        convd = np.abs(fiarr) > conv  # test for convergence
        nd = np.sum(convd is True)

    if Earr.size > 1:
        return Earr
    else:
        return Earr[0]