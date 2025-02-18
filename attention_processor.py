class RegionT2I_AttnProcessor:
    def __init__(self):
        print("[RegionT2I_AttnProcessor.__init__] Initializing processor")

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        print("[RegionT2I_AttnProcessor.__call__] Received kwargs:", kwargs)
        print("[RegionT2I_AttnProcessor.__call__] Region list:", kwargs.get('region_list'))
        # ... rest of the processor code 