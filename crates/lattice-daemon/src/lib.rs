use axum::{
    extract::{State, Json},
    response::sse::{Event, Sse},
    routing::{get, post},
    Router,
    http::StatusCode,
};
use futures_util::stream::{self, Stream};
use lattice_core::Achronon;
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, sync::Arc, time::Duration};
use tokio::sync::{broadcast, mpsc};
use tokio_stream::StreamExt;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum LatticeEvent {
    AionExpanded(Vec<Achronon>),
    AchrononPrecipitated(u32),
    StateUpdated(Vec<f32>),
    StabilityReached,
    Message(String),
}

#[derive(Debug, Deserialize)]
pub struct InjectRequest {
    pub content: String,
    pub antecedents: Vec<u32>,
    pub affected_subspace: Option<usize>,
}

pub struct DaemonState {
    pub tx: broadcast::Sender<LatticeEvent>,
    pub inject_tx: mpsc::Sender<InjectRequest>,
}

pub async fn run_daemon(
    tx: broadcast::Sender<LatticeEvent>,
    inject_tx: mpsc::Sender<InjectRequest>,
) -> anyhow::Result<()> {
    let state = Arc::new(DaemonState { tx, inject_tx });

    let app = Router::new()
        .route("/", get(index))
        .route("/stream", get(sse_handler))
        .route("/inject", post(inject_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000").await?;
    log::info!("Lattice Sandbox running at http://127.0.0.1:3000");
    axum::serve(listener, app).await?;

    Ok(())
}

async fn index() -> impl axum::response::IntoResponse {
    axum::response::Html(include_str!("../assets/index.html"))
}

async fn inject_handler(
    State(state): State<Arc<DaemonState>>,
    Json(payload): Json<InjectRequest>,
) -> StatusCode {
    match state.inject_tx.send(payload).await {
        Ok(_) => StatusCode::ACCEPTED,
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

async fn sse_handler(
    State(state): State<Arc<DaemonState>>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let rx = state.tx.subscribe();

    let stream = stream::unfold(rx, |mut rx| async move {
        match rx.recv().await {
            Ok(event) => {
                let sse_event = Event::default().json_data(event).unwrap();
                Some((Ok(sse_event), rx))
            }
            Err(_) => None,
        }
    });

    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(1))
            .text("keep-alive"),
    )
}
