"""Script to show how to enable optimaldeltat tracking"""
print("=" * 80)
print("HOW TO ENABLE optimaldeltat TRACKING")
print("=" * 80)

print("""
To enable optimaldeltat tracking, you need to set:
  options['trackOptimaldeltat'] = True

This should be set when trackUpstream is enabled.

The tracking will capture:
  - deltat (input time step)
  - varphimin
  - zetaP
  - rR (norm of Rt generators)
  - rerr1 (norm of Rerr generators)
  - varphiprod
  - deltats (candidate time steps)
  - objfuncset (objective function values)
  - bestIdxnew (selected index)
  - deltatest (selected time step)
  - kprimeest

This data will be stored in options['optimaldeltatLog'] and can be
compared between Python and MATLAB to find where the divergence occurs.
""")

print("\n" + "=" * 80)
print("COMPARING optimaldeltat IMPLEMENTATIONS")
print("=" * 80)

print("""
Key differences to check:

1. rR and rerr1 computation:
   Python: np.linalg.norm(np.sum(np.abs(Rt.generators()), axis=1), 2)
   MATLAB: vecnorm(sum(abs(generators(Rt)),2),2)
   
   These should be equivalent, but vecnorm behavior needs verification.

2. kprime indexing:
   Python: kprime = np.arange(0, kprimemax + 1)  # 0-indexed
   MATLAB: kprime = 0:round(kprimemax)  # 1-indexed array, but values 0,1,2,...
   
   Python: bestIdxnew is 0-indexed, kprimeest = kprime[bestIdxnew]
   MATLAB: bestIdxnew is 1-indexed, kprimeest = bestIdxnew - 1
   
   This should be handled correctly.

3. varphi computation:
   Python: varphi = (varphimin + (deltats[0] - deltats) / deltats[0] * varphi_h) / mu
   MATLAB: varphi = (varphimin + (deltats(1) - deltats)/deltats(1) * varphi_h) / mu
   
   Should be equivalent (deltats[0] vs deltats(1)).

4. sumallbutlast computation:
   Python: np.arange(1, int(floork[i]) + 1)  # 1 to floork[i] inclusive
   MATLAB: (1:floork(i))  # 1 to floork(i) inclusive
   
   Should be equivalent.

5. objfuncset computation:
   Python: rR * (1 + 2 * zetaZ) ** k * zetaP + rerr1 / k * varphiprod * (...)
   MATLAB: rR * (1+2*zetaZ) .^ k * zetaP + rerr1 ./ k .* varphiprod .* (...)
   
   Should be equivalent (element-wise operations).
""")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print("1. Enable trackOptimaldeltat in the tracking script")
print("2. Run both Python and MATLAB with tracking enabled")
print("3. Compare optimaldeltatLog entries for Step 2 Run 2")
print("4. Check if rR, rerr1, or other inputs differ")
print("5. Verify if objfuncset values match")
print("6. Check if bestIdxnew selection differs")
