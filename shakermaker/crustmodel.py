import numpy as np
from scipy.interpolate import interp1d

def interpolateme(x, y, xx, kind="previous"):
    return interp1d(x, y, fill_value=(y[0], y[-1]), bounds_error=False
        , kind=kind)(xx)

class CrustModel:

    def __init__(self, nlayers):
        self._d = np.zeros(nlayers)
        self._a = np.zeros(nlayers)
        self._b = np.zeros(nlayers)
        self._rho = np.zeros(nlayers)
        self._qa = np.zeros(nlayers)
        self._qb = np.zeros(nlayers)
        self._current_layer = 0
        self._nlayers = nlayers

    def add_layer(self, d, vp, vs, rho, gp, gs):
        assert self._current_layer <= self.nlayers, \
            "CrustModel.add_layer - current_layer={} Exceeds number of initialized " \
            "layers (nlayers={}).".format(self._current_layer, self.nlayers)

        self._d[self._current_layer] = d
        self._a[self._current_layer] = vp
        self._b[self._current_layer] = vs
        self._rho[self._current_layer] = rho
        self._qa[self._current_layer] = qp
        self._qb[self._current_layer] = qs

        self._current_layer += 1

    def modify_layer(self, layer_idx, d=None, vp=None, vs=None, rho=None, gp=None, gs=None):
        assert layer_idx >= self._current_layer, \
            "CrustModel.modify_layer - Exceeds number of initialized layers (nlayers={}). ".format(self._current_layer)

        if d is not None:
            self._d[layer_idx] = d
        if vp is not None:
            self._vp[layer_idx] = vp
        if vs is not None:
            self._vs[layer_idx] = vs
        if rho is not None:
            self._rho[layer_idx] = rho
        if gp is not None:
            self._gp[layer_idx] = gp
        if gs is not None:
            self._gs[layer_idx] = gs

    def properties_at_depths(self, z, kind="previous"):
        d = np.cumsum(self._d)
        a = interpolateme(d, self._a, z, kind)
        b = interpolateme(d, self._b, z, kind)
        rho = interpolateme(d, self._rho, z, kind)
        qa = interpolateme(d, self._qa, z, kind)
        qb = interpolateme(d, self._qb, z, kind)

        return a, b, rho, qa, qb

    def split_at_depth(self, z, tol=0.01):
        d = np.zeros(self.nlayers+1)
        a = np.zeros(self.nlayers+1)
        b = np.zeros(self.nlayers+1)
        rho = np.zeros(self.nlayers+1)
        qa = np.zeros(self.nlayers+1)
        qb = np.zeros(self.nlayers+1)

        zstart = 0
        zend = self._d[0]

        if zend == 0:
            zend += 1e10

        pos = 0
        was_split = False
        for i in range(self.nlayers):
            if (zstart+tol) <= z <= (zend-tol):
                d[pos] = z - zstart
                a[pos] = self._a[i]
                b[pos] = self._b[i]
                rho[pos] = self._rho[i]
                qa[pos] = self._qa[i]
                qb[pos] = self._qb[i]

                self._d[i] = max(self._d[i] - d[pos], 0)
                zstart = z
                pos += 1
                was_split = True

            d[pos] = self._d[i]
            a[pos] = self._a[i]
            b[pos] = self._b[i]
            rho[pos] = self._rho[i]
            qa[pos] = self._qa[i]
            qb[pos] = self._qb[i]
            pos += 1
            zstart += self._d[i]
            if self._d[i] == 0:
                break
            elif self._d[i+1] == 0:
                zend += 1e10
            else:
                zend += self.d[i+1]

        if was_split:
            self._d = d
            self._a = a
            self._b = b
            self._rho = rho
            self._qa = qa
            self._qb = qb
            self._nlayers += 1

    def get_layer(self, z, tol=0.01):
        current_z = 0
        for i in range(self.nlayers):
            current_z += self._d[i]
            if abs(z-current_z) < tol:
                return i
        return None

    @property
    def nlayers(self):
        return self._nlayers

    @property
    def d(self):
        return self._d

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def rho(self):
        return self._rho

    @property
    def qa(self):
        return self._qa

    @property
    def qb(self):
        return self._qb

    def __str__(self):
        """ Print a nice description of the current layer model

        """
        rep = "Crust Model with {} layers.\n".format(self.nlayers)
        #       12345678|12345678|12345678|12345678|12345678|12345678|12345678|
        rep += " Layer  | Depth  | Thick  | Vp     | Vs     | rho    | Qa     | Qb\n"
        fmt = "{0:8.0f}|{1:8.2f}|{2:8.2f}|{3:8.2f}|{4:8.2f}|{5:8.2f}|{6:8.1f}|{7:8.1f}"
        z = 0
        for i in range(self.nlayers):
            rep += fmt.format(i+1, z, self._d[i], self._a[i], self._b[i]
                , self._rho[i], self._qa[i], self._qb[i]) + "\n"
            z += self._d[i]

        return rep
