import re
p = "trace_rome.py"
s = open(p).read()

old_clean = '''        cache = {"h": {}, "mlp": {}, "attn": {}}
        handles = []
        for l, blk in enumerate(self.blocks):
            handles.append(blk.register_forward_hook(
                lambda m, i, o, l=l: cache["h"].__setitem__(l, o[0][0].clone())))
            handles.append(blk.mlp.register_forward_hook(
                lambda m, i, o, l=l: cache["mlp"].__setitem__(l, o[0].clone())))
            handles.append(blk.attn.register_forward_hook(
                lambda m, i, o, l=l: cache["attn"].__setitem__(l, (o[0] if isinstance(o, tuple) else o)[0].clone())))'''
new_clean = '''        cache = {"h": {}, "mlp": {}, "attn": {}}
        handles = []

        def saver(key, l):
            def hook(m, i, o):
                t = o[0] if isinstance(o, tuple) else o
                cache[key][l] = t[0].clone()
            return hook
        for l, blk in enumerate(self.blocks):
            handles.append(blk.register_forward_hook(saver("h", l)))
            handles.append(blk.mlp.register_forward_hook(saver("mlp", l)))
            handles.append(blk.attn.register_forward_hook(saver("attn", l)))'''
assert old_clean in s
s = s.replace(old_clean, new_clean)

old_mk = '''        def mk(l, rows, key, tuple_out):
            def hook(m, i, o):
                if tuple_out:
                    hid = o[0].clone()
                else:
                    hid = o.clone()
                for (r, t, src) in rows:
                    hid[r, t, :] = cache[key][src][t]
                if tuple_out:
                    return (hid,) + tuple(o[1:])
                return hid
            return hook'''
new_mk = '''        def mk(l, rows, key):
            def hook(m, i, o):
                is_tuple = isinstance(o, tuple)
                hid = (o[0] if is_tuple else o).clone()
                for (r, t, src) in rows:
                    hid[r, t, :] = cache[key][src][t]
                if is_tuple:
                    return (hid,) + tuple(o[1:])
                return hid
            return hook'''
assert old_mk in s
s = s.replace(old_mk, new_mk)

old_reg = '''            if component == "state":
                handles.append(self.blocks[l].register_forward_hook(mk(l, rows, "h", True)))
            elif component == "mlp":
                handles.append(self.blocks[l].mlp.register_forward_hook(mk(l, rows, "mlp", False)))
            else:
                handles.append(self.blocks[l].attn.register_forward_hook(mk(l, rows, "attn", True)))'''
new_reg = '''            if component == "state":
                handles.append(self.blocks[l].register_forward_hook(mk(l, rows, "h")))
            elif component == "mlp":
                handles.append(self.blocks[l].mlp.register_forward_hook(mk(l, rows, "mlp")))
            else:
                handles.append(self.blocks[l].attn.register_forward_hook(mk(l, rows, "attn")))'''
assert old_reg in s
s = s.replace(old_reg, new_reg)
open(p, "w").write(s)
print("ok")
