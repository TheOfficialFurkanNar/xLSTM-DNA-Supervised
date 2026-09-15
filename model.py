#model.py
import torch

class sLSTM(torch.nn.Module):
    def __init__(self, input_size,
                 hidden_size,
                 ):
        super(sLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        # Input gate layers
        self.input_gate_input = torch.nn.Linear(input_size, hidden_size)
        self.input_gate_hidden = torch.nn.Linear(hidden_size, hidden_size)
        
        # Forget gate layers
        self.forget_gate_input = torch.nn.Linear(input_size, hidden_size)
        self.forget_gate_hidden = torch.nn.Linear(hidden_size, hidden_size)
        
        # Output gate layers
        self.output_gate_input = torch.nn.Linear(input_size, hidden_size)
        self.output_gate_hidden = torch.nn.Linear(hidden_size, hidden_size)
        
        # Candidate cell state layers
        self.candidate_input = torch.nn.Linear(input_size, hidden_size)
        self.candidate_hidden = torch.nn.Linear(hidden_size, hidden_size)
        
        # Initialize forget gate bias to 1.0 for stable initial state
        torch.nn.init.constant_(self.forget_gate_input.bias, 1.0)
        torch.nn.init.constant_(self.forget_gate_hidden.bias, 0.0)
    
    def forward(self, x, state=None):
        if state is None:
            hidden_state_prev = torch.zeros(x.size(0), self.hidden_size, device=x.device)
            cell_state_prev = torch.zeros(x.size(0), self.hidden_size, device=x.device)
            stab_gate_prev = torch.zeros(x.size(0), self.hidden_size, device=x.device)
            normalizer_prev = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        else:
            hidden_state_prev, cell_state_prev, stab_gate_prev, normalizer_prev = state
        
        # Compute exponential gates (input and forget)
        log_input_gate = self.input_gate_input(x) + self.input_gate_hidden(hidden_state_prev)
        log_forget_gate = self.forget_gate_input(x) + self.forget_gate_hidden(hidden_state_prev)
        
        # Stabilizer gate for numerical stability
        stab_gate = torch.max(log_forget_gate + stab_gate_prev, log_input_gate)
        input_gate_stab = torch.exp(log_input_gate - stab_gate)
        forget_gate_stab = torch.exp(log_forget_gate + stab_gate_prev - stab_gate)
        
        # Use stabilized gates
        input_gate = input_gate_stab
        forget_gate = forget_gate_stab
        
        # Output gate (still uses sigmoid)
        output_gate = torch.sigmoid(self.output_gate_input(x) + self.output_gate_hidden(hidden_state_prev))
        
        # Compute candidate cell state
        cell_candidate = torch.tanh(self.candidate_input(x) + self.candidate_hidden(hidden_state_prev))
        
        # Cell state update with exponential gates
        cell_state_new = forget_gate * cell_state_prev + input_gate * cell_candidate
        
        # Normalizer state update
        normalizer_new = forget_gate * normalizer_prev + input_gate
        
        # Output computation with division by normalizer
        hidden_state_new = output_gate * (cell_state_new / (normalizer_new + 1e-8))
        
        return hidden_state_new, cell_state_new, stab_gate, normalizer_new


class mLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(mLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.head_dim = hidden_size  # For mLSTM, head_dim equals hidden_size
        
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
            normalizer_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
            stab_gate_prev = torch.zeros(batch_size, 1, device=x.device)
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
        
        # Forget gate pre-activation (eq. 26) - use sigmoid for forget gate
        log_forget_gate = torch.log(torch.sigmoid(self.forget_gate(x)))  # (batch, 1)
        
        # Stabilizer gate for numerical stability
        stab_gate = torch.max(log_forget_gate + stab_gate_prev, log_input_gate)
        input_gate_stab = torch.exp(log_input_gate - stab_gate)
        forget_gate_stab = torch.exp(log_forget_gate + stab_gate_prev - stab_gate)
        
        # Use stabilized gates (broadcast over matrix dimensions)
        input_gate = input_gate_stab  # (batch, 1)
        forget_gate = forget_gate_stab  # (batch, 1)
        
        # Outer product for cell state update (eq. 19): v_t * k_t^T
        # Use einsum for batched outer product: (batch, hidden) x (batch, hidden) -> (batch, hidden, hidden)
        cell_update = torch.einsum('bi,bj->bij', value_input, key_input)
        
        # Cell state update (eq. 19)
        cell_state_new = forget_gate.unsqueeze(-1) * cell_state_prev + input_gate.unsqueeze(-1) * cell_update
        
        # Normalizer state update (eq. 20)
        normalizer_new = forget_gate * normalizer_prev + input_gate * key_input
        
        # Output gate (eq. 27)
        output_gate = torch.sigmoid(self.output_gate(x))  # (batch, hidden)
        
        # Compute tilde h_t = C_t q_t / max(|n_t^T q_t|, 1) (eq. 21)
        # Matmul: (batch, hidden, hidden) x (batch, hidden) -> (batch, hidden)
        cell_query_product = torch.einsum('bij,bj->bi', cell_state_new, query_input)
        
        # Dot product for denominator: n_t^T q_t
        normalizer_query_dot = torch.einsum('bi,bi->b', normalizer_new, query_input)
        
        # Clamp denominator to avoid division by zero
        denominator = torch.max(torch.abs(normalizer_query_dot), torch.ones_like(normalizer_query_dot))
        
        # Compute tilde h_t
        tilde_hidden_state = cell_query_product / denominator.unsqueeze(-1)
        
        # Hidden state (eq. 21)
        hidden_state_new = output_gate * tilde_hidden_state
        
        return hidden_state_new, cell_state_new, normalizer_new, stab_gate
