import torch
import torch.nn as nn


class ResidualRiskBlock(nn.Module):

    def __init__(self, in_dim, n_upstream=0, hidden=64, h_dim=32, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.n_upstream = n_upstream
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, h_dim), nn.ReLU(),
        )
        self.residual_head = nn.Linear(h_dim, 1)   
        self.gate = nn.Linear(h_dim, 1)          
        self.flow = nn.Linear(n_upstream, 1, bias=False) if n_upstream > 0 else None
        if self.flow is not None:
            nn.init.constant_(self.flow.weight, 0.3) 
        nn.init.constant_(self.gate.bias, 2.0)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, x_g, prior_s, upstream=None,
                use_prior=True, use_flow=True, use_residual=True):

        h = self.encoder(x_g)
        if use_residual:
            r = self.residual_head(h).squeeze(-1)
            g = torch.sigmoid(self.gate(h)).squeeze(-1)
            residual = g * r
        else:
            residual = torch.zeros_like(prior_s)
        if use_flow and self.flow is not None and upstream is not None:
            flow = self.flow(upstream).squeeze(-1)    
        else:
            flow = torch.zeros_like(prior_s)
        base = prior_s if use_prior else torch.zeros_like(prior_s)
        risk = base + flow + residual                 
        return risk, residual, flow, h


class CIMR(nn.Module):
    ORDER = ["population", "internal", "history"]

    def __init__(self, group_dims, n_classes, hidden=64, h_dim=32, dropout=0.1,
                 use_prior=True, use_flow=True, use_residual=True):

        super().__init__()
        self.use_prior = use_prior
        self.use_flow = use_flow
        self.use_residual = use_residual
        self.blocks = nn.ModuleDict()
        for i, g in enumerate(self.ORDER):
            self.blocks[g] = ResidualRiskBlock(
                in_dim=group_dims[g], n_upstream=i,   
                hidden=hidden, h_dim=h_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(3 + h_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x_groups, priors):

        risks, residuals, flows, upstream_list = [], [], [], []
        h_last = None
        for i, g in enumerate(self.ORDER):
            upstream = torch.stack(upstream_list, dim=1) if upstream_list else None
            risk, res, flow, h = self.blocks[g](
                x_groups[g], priors[:, i], upstream,
                use_prior=self.use_prior, use_flow=self.use_flow,
                use_residual=self.use_residual)
            risks.append(risk)
            residuals.append(res)
            flows.append(flow)
            upstream_list.append(risk)     
            h_last = h
        risks_t = torch.stack(risks, dim=1)            
        residuals_t = torch.stack(residuals, dim=1)    
        flows_t = torch.stack(flows, dim=1)           
        fused = torch.cat([risks_t, h_last], dim=1)  
        logits = self.classifier(fused)
        return {"logits": logits, "risks": risks_t,
                "residuals": residuals_t, "flows": flows_t}

    def flow_coefficients(self):
        out = {}
        for i, g in enumerate(self.ORDER):
            blk = self.blocks[g]
            if blk.flow is not None:
                betas = blk.flow.weight.detach().cpu().numpy().ravel()
                out[g] = {self.ORDER[j]: float(betas[j]) for j in range(i)}
        return out
