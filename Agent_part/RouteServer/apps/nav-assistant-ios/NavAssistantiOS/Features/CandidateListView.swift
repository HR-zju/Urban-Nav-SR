import SwiftUI

struct CandidateListView: View {
    @ObservedObject var viewModel: NavigationFlowViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("候选列表")
                    .font(.title3.bold())
                Spacer()
                Button("语音播报") {
                    viewModel.announceCandidates()
                }
            }

            ForEach(viewModel.candidates) { candidate in
                Button {
                    Task { await viewModel.chooseCandidate(candidate) }
                } label: {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(candidate.title)
                            .font(.headline)
                            .foregroundStyle(.primary)
                        if !candidate.subtitle.isEmpty {
                            Text(candidate.subtitle)
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                        }
                        HStack(spacing: 12) {
                            if let distance = candidate.distanceText {
                                Label(distance, systemImage: "figure.walk")
                            }
                            if let duration = candidate.durationText {
                                Label(duration, systemImage: "clock")
                            }
                        }
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(14)
                    .background(
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .fill(viewModel.selectedCandidate == candidate ? Color.blue.opacity(0.12) : Color(uiColor: .secondarySystemBackground))
                    )
                }
                .buttonStyle(.plain)
            }
        }
        .padding(18)
        .background(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .fill(Color(uiColor: .systemBackground))
                .shadow(color: Color.black.opacity(0.06), radius: 14, y: 6)
        )
    }
}
