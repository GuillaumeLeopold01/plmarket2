import uuid, time, math, random, csv, statistics, warnings
from collections import deque, defaultdict
from datetime import datetime
from dataclasses import dataclass
from typing import List
try:
    from arch import arch_model
except ImportError:
    arch_model = None

@dataclass
class MarketConfig:
    garch_omega: float = 1e-6
    garch_alpha: float = 0.1
    garch_beta:  float = 0.85
    calibrate_period: int = 250
    history_window:  int = 800
    liquidity_base:  float = 1_000.0
    liquidity_coeff: float = 5e-4
    cvar_alpha:     float = 0.95
    cvar_limit:     float = 0.10
    price_k:        float = 10.0
    log_trades:     bool  = True
    log_file:       str   = "trades.csv"

class OddsProvider:
    def __init__(self):
        self.base = defaultdict(lambda: random.uniform(1.5, 5.0))
    def get_odds(self, team):
        self.base[team] = max(1.01, self.base[team]*(1+random.uniform(-.04,.04)))
        return self.base[team]

class GARCH:
    def __init__(self, ω, α, β, var0=1e-4):
        self.ω, self.α, self.β, self.var = ω, α, β, var0
    def update(self, r):
        self.var = self.ω + self.α*r*r + self.β*self.var
        return math.sqrt(self.var)
    def set_params(self, ω, α, β):
        self.ω, self.α, self.β = ω, α, β

class LiquidityModel:
    def __init__(self, base, coeff):
        self.base, self.coeff = base, coeff
    def impact(self, p, q):
        return p*(1+self.coeff*math.copysign(1,q)*abs(q)/self.base)

class PriceEngine:
    def __init__(self, cfg: MarketConfig):
        self.price, self.prev_prob, self.k = 1.0, None, cfg.price_k
        self.garch = GARCH(cfg.garch_omega, cfg.garch_alpha, cfg.garch_beta)
    def update(self, prob):
        if self.prev_prob is None:
            self.prev_prob = prob
            return self.price, 0.0
        r = prob - self.prev_prob
        self.prev_prob = prob
        vol = self.garch.update(r)
        self.price = max(.01, self.price*(1+self.k*r*(1+vol)))
        return self.price, vol

@dataclass
class Position:
    qty: float = 0.0
    cost: float = 0.0
    pnl: float = 0.0
    def avg(self): return self.cost/self.qty if self.qty else 0.0

class Participant:
    def __init__(self, name, cash=10_000.0):
        self.name, self.cash, self.pos = name, cash, Position()
        self.pnl_hist = deque(maxlen=600)

class Order:
    def __init__(self, side, qty, price, trader):
        self.id, self.side, self.qty = str(uuid.uuid4()), side, qty
        self.price, self.trader, self.time = price, trader, datetime.utcnow()

class OrderBook:
    def __init__(self): self.bids, self.asks = [], []
    def add(self,o):
        (self.bids if o.side=="buy" else self.asks).append(o)
        self.bids.sort(key=lambda x:(-x.price,x.time))
        self.asks.sort(key=lambda x:( x.price,x.time))
    def match(self):
        trades=[]
        while self.bids and self.asks and self.bids[0].price>=self.asks[0].price:
            b,s=self.bids[0],self.asks[0]; qty=min(b.qty,s.qty); p=(b.price+s.price)/2
            trades.append((b,s,qty,p)); b.qty-=qty; s.qty-=qty
            if b.qty==0: self.bids.pop(0)
            if s.qty==0: self.asks.pop(0)
        return trades

class CVaRRisk:
    def __init__(self,cfg:MarketConfig):
        self.a,self.lim=cfg.cvar_alpha,cfg.cvar_limit
    def ok(self,tr):
        losses=[-x for x in tr.pnl_hist if x<0]
        if len(losses)<10: return True
        losses.sort(); idx=int(len(losses)*self.a)
        cvar=statistics.mean(losses[idx:])
        return cvar<=self.lim*max(tr.cash,1)

class TeamMarket:
    def __init__(self, team, provider, cfg: MarketConfig):
        self.t, self.cfg, self.provider = team, cfg, provider
        self.engine, self.liq = PriceEngine(cfg), LiquidityModel(cfg.liquidity_base,cfg.liquidity_coeff)
        self.book, self.participants, self.risk = OrderBook(), set(), CVaRRisk(cfg)
        self.prob_hist: deque[float] = deque(maxlen=cfg.history_window)
        self.tick_count, self.last_cal = 0, 0
        if cfg.log_trades: self._init_log()
    def _init_log(self):
        with open(self.cfg.log_file,'w',newline='') as f:
            csv.writer(f).writerow(["ts","team","buy","sell","qty","price"])
    def _log(self,b,s,q,p):
        if not self.cfg.log_trades: return
        with open(self.cfg.log_file,'a',newline='') as f:
            csv.writer(f).writerow([datetime.utcnow(),self.t,b.trader.name,s.trader.name,q,p])
    def order(self,trader,side,qty):
        price,_=self.engine.price,0.0
        self.book.add(Order(side,qty,self.liq.impact(price,qty),trader))
        self.participants.add(trader)
    def _calibrate(self):
        if arch_model is None: return
        rets=[self.prob_hist[i]-self.prob_hist[i-1] for i in range(1,len(self.prob_hist))]
        if len(rets)<50: return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res=arch_model(rets,mean='Zero',vol='Garch',p=1,q=1).fit(disp='off')
        ω=res.params['omega']; α=res.params['alpha[1]']; β=res.params['beta[1]']
        if α+β<1 and all(x>0 for x in (ω,α,β)):
            self.engine.garch.set_params(ω,α,β)
    def tick(self):
        odds=self.provider.get_odds(self.t)
        prob=1/odds
        self.prob_hist.append(prob)
        price,vol=self.engine.update(prob)
        for b,s,q,p in self.book.match():
            cost=q*p; b.trader.cash-=cost; s.trader.cash+=cost
            b.trader.pos.qty+=q; s.trader.pos.qty-=q
            b.trader.pos.cost+=cost; s.trader.pos.cost-=cost
            self._log(b,s,q,p)
        for t in self.participants:
            t.pos.pnl=t.pos.qty*(price-t.pos.avg())
            t.pnl_hist.append(t.pos.pnl)
            if not self.risk.ok(t): self._liquidate(t)
        if abs(sum(p.pos.qty for p in self.participants))>1e-6:
            raise RuntimeError(f"Zero-sum breach on {self.t}")
        self.tick_count+=1
        if self.tick_count-self.last_cal>=self.cfg.calibrate_period:
            self._calibrate(); self.last_cal=self.tick_count
    def _liquidate(self,tr):
        if tr.pos.qty!=0:
            self.order(tr,"sell" if tr.pos.qty>0 else "buy",abs(tr.pos.qty))
            self.tick()
        tr.pos=Position()

class PremierLeagueMarket:
    TEAMS_2025=["Arsenal","Aston Villa","Bournemouth","Brentford","Brighton & Hove Albion",
                "Burnley","Chelsea","Crystal Palace","Everton","Fulham","Leeds United",
                "Liverpool","Manchester City","Manchester United","Newcastle United",
                "Nottingham Forest","Sunderland","Tottenham","West Ham","Wolverhampton"]
    def __init__(self,cfg=MarketConfig()):
        self.cfg,self.provider=cfg,OddsProvider()
        self.markets={t:TeamMarket(t,self.provider,cfg) for t in self.TEAMS_2025}
    def random_bot(self): return Participant(f"Bot{random.randint(1,9999)}")
    def step(self):
        m=random.choice(list(self.markets.values()))
        m.order(self.random_bot(),random.choice(["buy","sell"]),random.randint(1,150))
        for mk in self.markets.values(): mk.tick()

def demo(iters=1000,sleep=0.01):
    league=PremierLeagueMarket()
    for _ in range(iters):
        league.step(); time.sleep(sleep)

if __name__=="__main__":
    demo()
