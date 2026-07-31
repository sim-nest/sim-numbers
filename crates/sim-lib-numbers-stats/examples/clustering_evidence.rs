use sim_lib_numbers_stats::{
    CovarianceType, GmmControl, GmmSpec, KMeansControl, SingularComponentPolicy, fit_gmm,
    fit_kmeans,
};

fn main() {
    let points = vec![
        vec![-3.1, -3.0],
        vec![-3.0, -2.9],
        vec![-2.9, -3.1],
        vec![4.9, 5.0],
        vec![5.0, 5.1],
        vec![5.1, 4.9],
    ];
    let kmeans = fit_kmeans(
        &points,
        2,
        KMeansControl::new(4, 50, 1.0e-10, 20_000, 4).expect("control"),
    )
    .expect("bounded k-means");
    let selected = &kmeans.restarts[kmeans.selected_restart];
    println!(
        "kmeans centroids={:?} inertia={:.6} selected={} restarts={} converged={} work={}",
        kmeans.model.centroids,
        selected.inertia,
        kmeans.selected_restart,
        kmeans.restarts.len(),
        selected.converged,
        kmeans.work
    );

    let gmm = fit_gmm(
        &points,
        GmmSpec::new(
            2,
            CovarianceType::Diagonal,
            1.0e-6,
            SingularComponentPolicy::default(),
        )
        .expect("spec"),
        GmmControl::new(4, 50, 1.0e-9, 100_000).expect("control"),
    )
    .expect("regularized GMM");
    println!(
        "gmm means={:?} likelihood={:.6} iterations={} converged={} repairs={} aic={:.6} bic={:.6} work={}",
        gmm.model.means,
        gmm.evidence.log_likelihood,
        gmm.evidence.iterations,
        gmm.evidence.converged,
        gmm.evidence.singular_component_repairs,
        gmm.evidence.model_selection.aic,
        gmm.evidence.model_selection.bic,
        gmm.evidence.work
    );
}
