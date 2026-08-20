"""
Utilitarios geoespaciais: distancia, projecao metrica, referenciamento linear
sobre o tracado da linha e indexacao em celulas.

Por que referenciamento linear
------------------------------
Detectar chegada por "raio de N metros" erra feio em Sao Paulo: a rota costuma
passar duas vezes perto do mesmo ponto (ida e volta na mesma avenida, alcas,
retornos), e o GPS urbano tem erro de dezenas de metros. Projetando cada
posicao sobre a polilinha da rota obtemos a abscissa curvilinea `s` (metros
percorridos desde o inicio do tracado). A chegada na parada k vira um evento
simples e robusto: o instante em que `s(t)` cruza `s_k`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

RAIO_TERRA_M = 6371008.8


# --------------------------------------------------------------------- dist
def haversine_m(lat1, lon1, lat2, lon2):
    """Distancia em metros. Aceita escalares ou arrays numpy (broadcast)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * RAIO_TERRA_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def rumo_graus(lat1, lon1, lat2, lon2):
    """Azimute (0 = norte, sentido horario) do ponto 1 para o ponto 2."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


# ---------------------------------------------------------------- projecoes
class ProjecaoLocal:
    """
    Equirretangular local ancorada em (lat0, lon0).

    Sem dependencia externa. Na escala da Regiao Metropolitana de Sao Paulo
    (~100 km) o erro relativo de distancia fica abaixo de 0,1 %, ou seja menos
    de 10 cm em 100 m — muito abaixo do ruido do GPS, que e o que importa aqui.
    """

    nome = "equirretangular-local"

    def __init__(self, lat0: float = -23.5505, lon0: float = -46.6333):
        self.lat0 = lat0
        self.lon0 = lon0
        self._k_lat = math.radians(1.0) * RAIO_TERRA_M
        self._k_lon = math.radians(1.0) * RAIO_TERRA_M * math.cos(math.radians(lat0))

    def para_xy(self, lat, lon):
        x = (np.asarray(lon, dtype="float64") - self.lon0) * self._k_lon
        y = (np.asarray(lat, dtype="float64") - self.lat0) * self._k_lat
        return x, y

    def para_latlon(self, x, y):
        lon = np.asarray(x, dtype="float64") / self._k_lon + self.lon0
        lat = np.asarray(y, dtype="float64") / self._k_lat + self.lat0
        return lat, lon


class ProjecaoPyproj:
    """Projecao oficial (SIRGAS 2000 / UTM 23S = EPSG:31983) quando ha pyproj."""

    def __init__(self, epsg: int = 31983):
        from pyproj import Transformer

        self.epsg = epsg
        self.nome = f"EPSG:{epsg}"
        self._ida = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        self._volta = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)

    def para_xy(self, lat, lon):
        return self._ida.transform(np.asarray(lon, "float64"), np.asarray(lat, "float64"))

    def para_latlon(self, x, y):
        lon, lat = self._volta.transform(np.asarray(x, "float64"), np.asarray(y, "float64"))
        return lat, lon


def obter_projecao(epsg: int = 31983, lat0: float = -23.5505, lon0: float = -46.6333):
    """Usa pyproj se estiver instalado; senao cai no fallback interno."""
    try:
        return ProjecaoPyproj(epsg)
    except Exception:
        return ProjecaoLocal(lat0, lon0)


# ------------------------------------------------------- referenciamento linear
@dataclass
class ResultadoProjecao:
    s: float          # abscissa curvilinea, metros desde o inicio do tracado
    distancia: float  # distancia perpendicular ao tracado, metros
    segmento: int     # indice do segmento onde caiu


class ReferenciadorLinear:
    """
    Projeta pontos sobre uma polilinha ja em coordenadas metricas.

    Uso:
        ref = ReferenciadorLinear(xs, ys)
        s, dist, seg = ref.projetar(x, y)
    """

    def __init__(self, xs: Sequence[float], ys: Sequence[float]):
        pts = np.column_stack(
            [np.asarray(xs, dtype="float64"), np.asarray(ys, dtype="float64")]
        )
        if len(pts) < 2:
            raise ValueError("tracado precisa de pelo menos 2 pontos")
        self.pts = pts
        self.a = pts[:-1]
        self.v = np.diff(pts, axis=0)
        self.len2 = np.einsum("ij,ij->i", self.v, self.v)
        self.len2[self.len2 == 0.0] = 1e-9
        comprimentos = np.sqrt(self.len2)
        self.s_no = np.concatenate([[0.0], np.cumsum(comprimentos)])
        self.comprimento = float(self.s_no[-1])

    # -- projecao de um ponto ------------------------------------------------
    def projetar(
        self, x: float, y: float, janela: tuple[float, float] | None = None
    ) -> ResultadoProjecao:
        idx = np.arange(len(self.a))
        if janela is not None:
            s0, s1 = janela
            mask = (self.s_no[1:] >= s0) & (self.s_no[:-1] <= s1)
            if mask.any():
                idx = idx[mask]

        a = self.a[idx]
        v = self.v[idx]
        len2 = self.len2[idx]

        w = np.array([x, y], dtype="float64") - a
        t = np.clip(np.einsum("ij,ij->i", w, v) / len2, 0.0, 1.0)
        proj = a + t[:, None] * v
        d = np.hypot(proj[:, 0] - x, proj[:, 1] - y)

        k = int(np.argmin(d))
        seg = int(idx[k])
        s = float(self.s_no[seg] + t[k] * math.sqrt(self.len2[seg]))
        return ResultadoProjecao(s=s, distancia=float(d[k]), segmento=seg)

    # -- projecao de uma trajetoria inteira ----------------------------------
    def projetar_sequencia(
        self,
        xs: Sequence[float],
        ys: Sequence[float],
        janela_atras_m: float = 800.0,
        janela_frente_m: float = 8000.0,
        dist_aceitavel_m: float = 150.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Projeta uma sequencia temporal usando o `s` anterior como ancora.

        Isso resolve o caso do tracado que passa duas vezes perto do mesmo
        lugar: entre dois candidatos, escolhe o coerente com o que ja foi
        percorrido. Se a melhor projecao dentro da janela ficar longe demais,
        refaz sem janela (o veiculo pode ter saido e voltado ao itinerario).
        """
        xs = np.asarray(xs, dtype="float64")
        ys = np.asarray(ys, dtype="float64")
        n = len(xs)
        s_out = np.empty(n, dtype="float64")
        d_out = np.empty(n, dtype="float64")

        s_ant: float | None = None
        for i in range(n):
            janela = (
                None
                if s_ant is None
                else (s_ant - janela_atras_m, s_ant + janela_frente_m)
            )
            r = self.projetar(xs[i], ys[i], janela)
            if janela is not None and r.distancia > dist_aceitavel_m:
                alternativa = self.projetar(xs[i], ys[i], None)
                if alternativa.distancia < r.distancia * 0.6:
                    r = alternativa
            s_out[i] = r.s
            d_out[i] = r.distancia
            s_ant = r.s
        return s_out, d_out

    def ponto_em(self, s: float) -> tuple[float, float]:
        """Coordenada metrica na abscissa `s`."""
        s = float(np.clip(s, 0.0, self.comprimento))
        seg = int(np.searchsorted(self.s_no, s, side="right") - 1)
        seg = max(0, min(seg, len(self.a) - 1))
        resto = s - self.s_no[seg]
        t = resto / math.sqrt(self.len2[seg])
        p = self.a[seg] + t * self.v[seg]
        return float(p[0]), float(p[1])


# ------------------------------------------------------------------ tracados
def simplificar(xs, ys, tolerancia_m: float = 8.0):
    """Douglas-Peucker iterativo (sem recursao, aguenta trajetorias longas)."""
    pts = np.column_stack([np.asarray(xs, "float64"), np.asarray(ys, "float64")])
    n = len(pts)
    if n < 3:
        return pts[:, 0], pts[:, 1]

    manter = np.zeros(n, dtype=bool)
    manter[0] = manter[-1] = True
    pilha = [(0, n - 1)]
    while pilha:
        i, j = pilha.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        v = b - a
        len2 = float(v @ v) or 1e-9
        w = pts[i + 1 : j] - a
        t = np.clip((w @ v) / len2, 0.0, 1.0)
        proj = a + t[:, None] * v
        d = np.hypot(proj[:, 0] - pts[i + 1 : j, 0], proj[:, 1] - pts[i + 1 : j, 1])
        k = int(np.argmax(d))
        if d[k] > tolerancia_m:
            corte = i + 1 + k
            manter[corte] = True
            pilha.append((i, corte))
            pilha.append((corte, j))
    return pts[manter, 0], pts[manter, 1]


def densificar(xs, ys, passo_m: float = 25.0):
    """Insere pontos intermediarios para o referenciamento ficar mais fino."""
    xs = np.asarray(xs, "float64")
    ys = np.asarray(ys, "float64")
    out_x: list[float] = []
    out_y: list[float] = []
    for i in range(len(xs) - 1):
        d = math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
        n = max(1, int(d // passo_m))
        for k in range(n):
            f = k / n
            out_x.append(xs[i] + f * (xs[i + 1] - xs[i]))
            out_y.append(ys[i] + f * (ys[i + 1] - ys[i]))
    out_x.append(float(xs[-1]))
    out_y.append(float(ys[-1]))
    return np.array(out_x), np.array(out_y)


# ------------------------------------------------------------------- celulas
def celula(lat: float, lon: float, resolucao: int = 9) -> str:
    """
    Identificador de celula para agregacao regional.

    Usa H3 se disponivel (hexagonos, sem distorcao de area), senao cai numa
    grade regular equivalente em metros — o suficiente para agrupar hotspots.
    """
    try:
        import h3

        if hasattr(h3, "latlng_to_cell"):  # h3 >= 4
            return h3.latlng_to_cell(lat, lon, resolucao)
        return h3.geo_to_h3(lat, lon, resolucao)  # h3 3.x
    except Exception:
        aresta = {7: 1220.0, 8: 461.0, 9: 174.0, 10: 66.0}.get(resolucao, 174.0)
        proj = _PROJ_PADRAO
        x, y = proj.para_xy(lat, lon)
        return f"g{resolucao}_{int(float(x) // aresta)}_{int(float(y) // aresta)}"


def celulas(lats, lons, resolucao: int = 9) -> list[str]:
    return [celula(float(a), float(b), resolucao) for a, b in zip(lats, lons)]


def centro_celula(cel: str) -> tuple[float, float] | None:
    """(lat, lon) do centro da celula, para desenhar o mapa depois."""
    if cel.startswith("g") and cel.count("_") == 2:
        prefixo, sx, sy = cel.split("_")
        try:
            res = int(prefixo[1:])
            ix, iy = int(sx), int(sy)
        except ValueError:
            return None
        aresta = {7: 1220.0, 8: 461.0, 9: 174.0, 10: 66.0}.get(res, 174.0)
        lat, lon = _PROJ_PADRAO.para_latlon(
            (ix + 0.5) * aresta, (iy + 0.5) * aresta
        )
        return float(lat), float(lon)
    try:
        import h3

        if hasattr(h3, "cell_to_latlng"):
            return h3.cell_to_latlng(cel)
        return h3.h3_to_geo(cel)
    except Exception:
        return None


_PROJ_PADRAO = ProjecaoLocal()


# ------------------------------------------------------------------- geojson
def ponto_geojson(lat: float, lon: float) -> dict:
    """GeoJSON Point no formato que o MongoDB indexa com 2dsphere."""
    return {"type": "Point", "coordinates": [float(lon), float(lat)]}


def linha_geojson(lats: Iterable[float], lons: Iterable[float]) -> dict:
    return {
        "type": "LineString",
        "coordinates": [[float(x), float(y)] for y, x in zip(lats, lons)],
    }
