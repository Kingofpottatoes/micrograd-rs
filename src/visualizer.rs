use crate::autograd::Var;
use crate::nn::Module;
use crate::tensor::Tensor;
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

    chart
        .configure_mesh()
        .x_desc("step")
        .y_desc("loss")
        .draw()?;

    chart.draw_series(LineSeries::new(
        losses.iter().enumerate().map(|(i, &l)| (i, l)),
        &BLUE.mix(0.3),
    ))?;
    let means = moving_average(losses, 150);

    chart.draw_series(LineSeries::new(
        means.iter().enumerate().map(|(i, &l)| (i, l)),
        &RED,
    ))?;

    root.present()?;
    Ok(())
}

pub fn moving_average(values: &[f32], window: usize) -> Vec<f32> {
    assert!(window > 0, "Value error, window : 0 ");
    let mut means = Vec::new();
    for i in 0..values.len() {
        let start = i.saturating_sub(window - 1);
        let slice = &values[start..=i];
        let mean = slice.iter().sum::<f32>() / (slice.len() as f32);
        means.push(mean);
    }
    means
}

pub fn make_grid(x_min: f32, x_max: f32, y_min: f32, y_max: f32, res: usize) -> Tensor {
    let step_x = (x_max - x_min) / res as f32;
    let step_y = (y_max - y_min) / res as f32;
    let mut points = Vec::new();
    for i in 0..res {
        for j in 0..res {
            let x = x_min + (i as f32) * step_x;
            let y = y_min + (j as f32) * step_y;
            points.push(x);
            points.push(y);
        }
    }
    Tensor::from_vec(points, res * res, 2)
}

pub fn plot_decision_boundary(
    model: &dyn Module,
    points: &Tensor,
    labels: &[usize],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(points.cols(), 2);

    let root = BitMapBackend::new(path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let x_min = points
        .data()
        .iter()
        .step_by(2)
        .cloned()
        .fold(f32::INFINITY, f32::min)
        - 0.5;
    let x_max = points
        .data()
        .iter()
        .step_by(2)
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max)
        + 0.5;

    let y_min = points
        .data()
        .iter()
        .skip(1)
        .step_by(2)
        .cloned()
        .fold(f32::INFINITY, f32::min)
        - 0.5;
    let y_max = points
        .data()
        .iter()
        .skip(1)
        .step_by(2)
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max)
        + 0.5;

    let res = 200;

    let step_x = (x_max - x_min) / res as f32;
    let step_y = (y_max - y_min) / res as f32;

    let grid = make_grid(x_min, x_max, y_min, y_max, res);
    let output = model.forward(&Var::new(grid));
    let classes = output.data().argmax_rows();

    let mut chart = ChartBuilder::on(&root)
        .caption("Decision boundary", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart.configure_mesh().x_desc("x").y_desc("y").draw()?;

    chart.draw_series((0..res * res).map(|k| {
        let i = k / res;
        let j = k % res;
        let corner1 = (x_min + (i as f32) * step_x, y_min + (j as f32) * step_y);
        let corner2 = (
            x_min + ((i + 1) as f32) * step_x,
            y_min + ((j + 1) as f32) * step_y,
        );
        Rectangle::new(
            [corner1, corner2],
            Palette99::pick(classes[k]).mix(0.3).filled(),
        )
    }))?;

    chart.draw_series((0..points.rows()).map(|r| {
        Circle::new(
            (points.get(r, 0), points.get(r, 1)),
            4,
            Palette99::pick(labels[r]).filled(),
        )
    }))?;

    root.present()?;

    Ok(())
}
