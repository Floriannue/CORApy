"""Compare Step 2's MATLAB entry to see what tracking data is available"""
import numpy as np
import scipy.io as sio

print("=" * 80)
print("ANALYZING STEP 2 MATLAB ENTRY")
print("=" * 80)

# Load MATLAB log
matlab_file = 'upstream_matlab_log.mat'
matlab_data = sio.loadmat(matlab_file, squeeze_me=False, struct_as_record=False)
matlab_upstream = matlab_data.get('upstreamLog', [])

def get_ml_value(ml_obj, field):
    if hasattr(ml_obj, 'dtype') and ml_obj.dtype.names and field in ml_obj.dtype.names:
        val = ml_obj[field]
        if isinstance(val, np.ndarray) and val.size > 0:
            if val.dtype == object and val.size == 1:
                val = val.item()
            elif hasattr(val, 'dtype') and val.dtype == object:
                val = val.item() if val.size == 1 else val[0,0]
        return val
    elif hasattr(ml_obj, field):
        val = getattr(ml_obj, field)
        if isinstance(val, np.ndarray) and val.size > 0:
            if val.dtype == object or (hasattr(val, 'dtype') and val.dtype == object):
                val = val.item() if val.size == 1 else val[0,0]
            elif val.size == 1:
                val = val.item()
        return val
    return None

# Find Step 2 entry
step2 = 2
ml_step2 = None
if isinstance(matlab_upstream, np.ndarray) and matlab_upstream.ndim == 2:
    for i in range(matlab_upstream.shape[0]):
        e = matlab_upstream[i, 0]
        step_val = get_ml_value(e, 'step')
        if step_val is not None:
            s = step_val.item() if isinstance(step_val, np.ndarray) and step_val.size == 1 else step_val[0,0] if isinstance(step_val, np.ndarray) and step_val.size > 0 else step_val
            if s == step2:
                ml_step2 = e
                break

if ml_step2:
    print(f"\nStep 2 MATLAB Entry Fields:")
    if hasattr(ml_step2, 'dtype') and ml_step2.dtype.names:
        print(f"  All fields: {list(ml_step2.dtype.names)}")
    
    # Check Rtp_final_tracking - use getattr for mat_struct
    print("\n1. Rtp_final_tracking:")
    if hasattr(ml_step2, 'Rtp_final_tracking'):
        rtp_final = ml_step2.Rtp_final_tracking
        print(f"   Type: {type(rtp_final)}")
        if isinstance(rtp_final, np.ndarray):
            print(f"   Shape: {rtp_final.shape}, Size: {rtp_final.size}, Dtype: {rtp_final.dtype}")
            if rtp_final.size > 0:
                if rtp_final.dtype == object:
                    rtp_final = rtp_final.item() if rtp_final.size == 1 else rtp_final[0,0]
                    print(f"   After extraction type: {type(rtp_final)}")
        
        if rtp_final is not None and not (isinstance(rtp_final, np.ndarray) and rtp_final.size == 0):
            try:
                num_gen = getattr(rtp_final, 'num_generators', None)
                timeStepequalHorizon = getattr(rtp_final, 'timeStepequalHorizon_used', None)
                print(f"   num_generators attr: {num_gen}, type: {type(num_gen)}")
                print(f"   timeStepequalHorizon_used attr: {timeStepequalHorizon}, type: {type(timeStepequalHorizon)}")
                if num_gen is not None:
                    if isinstance(num_gen, np.ndarray):
                        num_gen = num_gen.item() if num_gen.size == 1 else num_gen[0,0] if num_gen.size > 0 else None
                    print(f"   num_generators: {num_gen}")
                if timeStepequalHorizon is not None:
                    if isinstance(timeStepequalHorizon, np.ndarray):
                        timeStepequalHorizon = timeStepequalHorizon.item() if timeStepequalHorizon.size == 1 else timeStepequalHorizon[0,0] if timeStepequalHorizon.size > 0 else None
                    print(f"   timeStepequalHorizon_used: {timeStepequalHorizon}")
                    if timeStepequalHorizon:
                        print("   → Step 2 uses timeStepequalHorizon path!")
                    else:
                        print("   → Step 2 uses normal path")
            except Exception as e:
                print(f"   Error extracting values: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("   Field is None or empty array")
    else:
        print("   Field not found")
    
    # Check Rlintp_tracking - use getattr for mat_struct
    print("\n2. Rlintp_tracking:")
    if hasattr(ml_step2, 'Rlintp_tracking'):
        rlintp = ml_step2.Rlintp_tracking
        if isinstance(rlintp, np.ndarray) and rlintp.size > 0:
            if rlintp.dtype == object:
                rlintp = rlintp.item() if rlintp.size == 1 else rlintp[0,0]
        
        if rlintp is not None:
            try:
                num_gen = getattr(rlintp, 'num_generators', None)
                if num_gen is not None:
                    if isinstance(num_gen, np.ndarray):
                        num_gen = num_gen.item() if num_gen.size == 1 else num_gen[0,0] if num_gen.size > 0 else None
                    print(f"   num_generators: {num_gen}")
            except Exception as e:
                print(f"   Error extracting values: {e}")
    else:
        print("   Field not found")
    
    # Check Rerror_tracking - use getattr for mat_struct
    print("\n3. Rerror_tracking:")
    if hasattr(ml_step2, 'Rerror_tracking'):
        rerror = ml_step2.Rerror_tracking
        if isinstance(rerror, np.ndarray) and rerror.size > 0:
            if rerror.dtype == object:
                rerror = rerror.item() if rerror.size == 1 else rerror[0,0]
        
        if rerror is not None:
            try:
                num_gen = getattr(rerror, 'num_generators', None)
                if num_gen is not None:
                    if isinstance(num_gen, np.ndarray):
                        num_gen = num_gen.item() if num_gen.size == 1 else num_gen[0,0] if num_gen.size > 0 else None
                    print(f"   num_generators: {num_gen}")
            except Exception as e:
                print(f"   Error extracting values: {e}")
    else:
        print("   Field not found")
    
    # Check Rtp_h_tracking (if timeStepequalHorizon path) - use getattr for mat_struct
    print("\n4. Rtp_h_tracking:")
    if hasattr(ml_step2, 'Rtp_h_tracking'):
        rtp_h = ml_step2.Rtp_h_tracking
        if isinstance(rtp_h, np.ndarray) and rtp_h.size > 0:
            if rtp_h.dtype == object:
                rtp_h = rtp_h.item() if rtp_h.size == 1 else rtp_h[0,0]
        
        if rtp_h is not None:
            try:
                num_gen = getattr(rtp_h, 'num_generators', None)
                if num_gen is not None:
                    if isinstance(num_gen, np.ndarray):
                        num_gen = num_gen.item() if num_gen.size == 1 else num_gen[0,0] if num_gen.size > 0 else None
                    print(f"   num_generators: {num_gen}")
                    print("   → This is Rtp_h from Step 1 (becomes Rtp in Step 2)")
            except Exception as e:
                print(f"   Error extracting values: {e}")
    else:
        print("   Field not found")
else:
    print("Step 2 entry not found")
