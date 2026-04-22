use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "realtime".to_string());

    match mode.as_str() {
        "realtime" => {
            let steps = args.next().and_then(|v| v.parse().ok()).unwrap_or(600);
            sim_cli::realtime_entrypoint(steps)
        }
        "fixed-step" => {
            let steps = args.next().and_then(|v| v.parse().ok()).unwrap_or(600);
            let dt_seconds = args
                .next()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1.0 / 120.0);
            sim_cli::fixed_step_entrypoint(steps, dt_seconds)
        }
        _ => {
            eprintln!("Usage: sim-cli [realtime [steps] | fixed-step [steps] [dt_seconds]]");
            Ok(())
        }
    }
}
