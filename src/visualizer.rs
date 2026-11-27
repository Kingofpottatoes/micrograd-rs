use plotters::prelude::*;

pub fn plot_loss(losses: &[f32], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let min = losses.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = losses.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Training loss", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0..losses.len(), (min..max).log_scale())?;

    chart.configure_mesh().x_desc("step").y_desc("loss").draw()?;

    chart.draw_series(LineSeries::new(
        losses.iter().enumerate().map(|(i, &l)| (i, l)),
        &BLUE,
    ))?;

    root.present()?;
    Ok(())
}