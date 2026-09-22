#model.py
import torch

class sLSTM(torch.nn.Module):
    def __init__(self, input_size,
                 hidden_size,
                 use_exp_gating=True,
                 use_stabilizer=True,
                 use_normalizer=True,
                 use_memory_mixing=True
                 ):
        super(sLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.use_exp_gating = use_exp_gating
        self.use_stabilizer = use_stabilizer
        self.use_normalizer = use_normalizer
        self.use_memory_mixing = use_memory_mixing
        
        # Input gate layers
        self.input_gate_input = torch.nn.Linear(input_size, hidden_size)
        self.input_gate_hidden = torch.nn.Linear(hidden_size, hidden_size) if use_memory_mixing else None
        
        # Forget gate layers
        self.forget_gate_input = torch.nn.Linear(input_size, hidden_size)
        self.forget_gate_hidden = torch.nn.Linear(hidden_size, hidden_size) if use_memory_mixing else None
        
        # Output gate layers
        self.output_gate_input = torch.nn.Linear(input_size, hidden_size)
        self.output_gate_hidden = torch.nn.Linear(hidden_size, hidden_size) if use_memory_mixing else None
        
        # Candidate cell state layers
        self.candidate_input = torch.nn.Linear(input_size, hidden_size)
        self.candidate_hidden = torch.nn.Linear(hidden_size, hidden_size) if use_memory_mixing else None
        
        # Initialize forget gate bias to 1.0 for stable initial state
        torch.nn.init.constant_(self.forget_gate_input.bias, 1.0)
        if use_memory_mixing:
            torch.nn.init.constant_(self.forget_gate_hidden.bias, 0.0)
    
    def forward(self, x, state=None):
        batch_size = x.size(0)
        
        if state is None:
            hidden_state_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
            cell_state_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
            stab_gate_prev = torch.zeros(batch_size, self.hidden_size, device=x.device) if self.use_stabilizer else None
            normalizer_prev = torch.zeros(batch_size, self.hidden_size, device=x.device) if self.use_normalizer else None
        else:
            hidden_state_prev, cell_state_prev, stab_gate_prev, normalizer_prev = state
        
        # Compute gate inputs
        if self.use_memory_mixing:
            log_input_gate = self.input_gate_input(x) + self.input_gate_hidden(hidden_state_prev)
            log_forget_gate = self.forget_gate_input(x) + self.forget_gate_hidden(hidden_state_prev)
            output_gate_input = self.output_gate_input(x) + self.output_gate_hidden(hidden_state_prev)
            cell_candidate_input = self.candidate_input(x) + self.candidate_hidden(hidden_state_prev)
        else:
            log_input_gate = self.input_gate_input(x)
            log_forget_gate = self.forget_gate_input(x)
            output_gate_input = self.output_gate_input(x)
            cell_candidate_input = self.candidate_input(x)
        
        # Apply exponential or sigmoid gating
        if self.use_exp_gating:
            if self.use_stabilizer:
                stab_gate = torch.max(log_forget_gate + stab_gate_prev, log_input_gate)
                input_gate = torch.exp(log_input_gate - stab_gate)
                forget_gate = torch.exp(log_forget_gate + stab_gate_prev - stab_gate)
            else:
                input_gate = torch.exp(log_input_gate)
                forget_gate = torch.exp(log_forget_gate)
        else:
            input_gate = torch.sigmoid(log_input_gate)
            forget_gate = torch.sigmoid(log_forget_gate)
        
        # Output gate (always sigmoid)
        output_gate = torch.sigmoid(output_gate_input)
        
        # Candidate cell state
        cell_candidate = torch.tanh(cell_candidate_input)
        
        # Cell state update
        cell_state_new = forget_gate * cell_state_prev + input_gate * cell_candidate
        
        # Normalizer state update
        if self.use_normalizer:
            normalizer_new = forget_gate * normalizer_prev + input_gate
        else:
            normalizer_new = None
        
        # Output computation
        if self.use_normalizer:
            hidden_state_new = output_gate * (cell_state_new / (normalizer_new + 1e-8))
        else:
            hidden_state_new = output_gate * cell_candidate
        
        # Return stabilizer if needed
        if self.use_stabilizer and self.use_exp_gating:
            return hidden_state_new, cell_state_new, stab_gate, normalizer_new
        else:
            return hidden_state_new, cell_state_new, None, normalizer_new


class mLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size,
                 use_exp_gating=True,
                 use_stabilizer=True,
                 use_normalizer=True
                 ):
        super(mLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.head_dim = hidden_size
        self.use_exp_gating = use_exp_gating
        self.use_stabilizer = use_stabilizer
        self.use_normalizer = use_normalizer
        
        # Query projection (eq. 22)
        self.query_proj = torch.nn.Linear(input_size, hidden_size)
        
        # Key projection (eq. 23) - scaled by 1/sqrt(d)
        self.key_proj = torch.nn.Linear(input_size, hidden_size)
        
        # Value projection (eq. 24)
        self.value_proj = torch.nn.Linear(input_size, hidden_size)
        
        # Input gate (eq. 25) - scalar gate
        self.input_gate = torch.nn.Linear(input_size, 1)
        
        # Forget gate (eq. 26) - scalar gate
        self.forget_gate = torch.nn.Linear(input_size, 1)
        
        # Output gate (eq. 27)
        self.output_gate = torch.nn.Linear(input_size, hidden_size)
        
        # Initialize forget gate bias to 1.0 for stable initial state
        torch.nn.init.constant_(self.forget_gate.bias, 1.0)
    
    def forward(self, x:torch.Tensor, state=None):
        """
        Forward pass of mLSTM with matrix memory and stabilizer gates
        Args:
            x: Input tensor of shape (batch_size, input_size)
            state: Tuple of (cell_state_prev, normalizer_prev, stab_gate_prev) from previous timestep
        Returns:
            hidden_state_new: New hidden state
            cell_state_new: New cell state (matrix)
            normalizer_new: New normalizer state
            stab_gate: New stabilizer gate
        """
        batch_size = x.size(0)
        
        if state is None:
            cell_state_prev = torch.zeros(batch_size, self.hidden_size, self.hidden_size, device=x.device)
            normalizer_prev = torch.zeros(batch_size, self.hidden_size, device=x.device) if self.use_normalizer else None
            stab_gate_prev = torch.zeros(batch_size, 1, device=x.device) if self.use_stabilizer else None
        else:
            cell_state_prev, normalizer_prev, stab_gate_prev = state
        
        # Query projection (eq. 22)
        query_input = self.query_proj(x)  # (batch, hidden)
        
        # Key projection with 1/sqrt(d) scaling (eq. 23)
        key_input = self.key_proj(x) / (self.head_dim ** 0.5)  # (batch, hidden)
        
        # Value projection (eq. 24)
        value_input = self.value_proj(x)  # (batch, hidden)
        
        # Input gate pre-activation (eq. 25)
        log_input_gate = self.input_gate(x)  # (batch, 1)
        
        # Forget gate pre-activation (eq. 26)
        if self.use_exp_gating:
            log_forget_gate = self.forget_gate(x)
        else:
            log_forget_gate = torch.log(torch.sigmoid(self.forget_gate(x)))
        
        # Apply stabilizer if enabled
        if self.use_exp_gating and self.use_stabilizer:
            stab_gate = torch.max(log_forget_gate + stab_gate_prev, log_input_gate)
            input_gate = torch.exp(log_input_gate - stab_gate)
            forget_gate = torch.exp(log_forget_gate + stab_gate_prev - stab_gate)
        elif self.use_exp_gating:
            input_gate = torch.exp(log_input_gate)
            forget_gate = torch.exp(log_forget_gate)
        else:
            input_gate = torch.sigmoid(log_input_gate)
            forget_gate = torch.sigmoid(log_forget_gate)
        
        # Outer product for cell state update (eq. 19): v_t * k_t^T
        cell_update = torch.einsum('bi,bj->bij', value_input, key_input)
        
        # Cell state update (eq. 19)
        cell_state_new = forget_gate.unsqueeze(-1) * cell_state_prev + input_gate.unsqueeze(-1) * cell_update
        
        # Normalizer state update (eq. 20)
        if self.use_normalizer:
            normalizer_new = forget_gate * normalizer_prev + input_gate * key_input
        else:
            normalizer_new = None
        
        # Output gate (eq. 27)
        output_gate = torch.sigmoid(self.output_gate(x))  # (batch, hidden)
        
        # Compute tilde h_t = C_t q_t / max(|n_t^T q_t|, 1) (eq. 21)
        cell_query_product = torch.einsum('bij,bj->bi', cell_state_new, query_input)
        
        if self.use_normalizer:
            normalizer_query_dot = torch.einsum('bi,bi->b', normalizer_new, query_input)
            denominator = torch.max(torch.abs(normalizer_query_dot), torch.ones_like(normalizer_query_dot))
            tilde_hidden_state = cell_query_product / denominator.unsqueeze(-1)
        else:
            tilde_hidden_state = cell_query_product / (self.hidden_size ** 0.5)
        
        # Hidden state (eq. 21)
        hidden_state_new = output_gate * tilde_hidden_state
        
        # Return stabilizer if needed
        if self.use_stabilizer and self.use_exp_gating:
            return hidden_state_new, cell_state_new, normalizer_new, stab_gate
        else:
            return hidden_state_new, cell_state_new, normalizer_new, None
