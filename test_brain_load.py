
try:
    from ouroboros.brain import Brain
    print("SUCCESS: Brain is loadable")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
