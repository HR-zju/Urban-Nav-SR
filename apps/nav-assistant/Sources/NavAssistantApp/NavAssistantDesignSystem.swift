import SwiftUI
import AppKit

enum NavSurfaceTone {
    case neutral
    case accent
    case success
    case warning
    case danger

    var tint: Color {
        switch self {
        case .neutral:
            return Color.accentColor
        case .accent:
            return Color.accentColor
        case .success:
            return Color(nsColor: .systemGreen)
        case .warning:
            return Color(nsColor: .systemOrange)
        case .danger:
            return Color(nsColor: .systemRed)
        }
    }

    var symbol: String {
        switch self {
        case .neutral:
            return "circle.fill"
        case .accent:
            return "sparkles"
        case .success:
            return "checkmark.circle.fill"
        case .warning:
            return "exclamationmark.triangle.fill"
        case .danger:
            return "xmark.octagon.fill"
        }
    }
}

extension Color {
    static let navWindow = Color(nsColor: .windowBackgroundColor)
    static let navCardBackground = Color(nsColor: .controlBackgroundColor)
    static let navRaisedBackground = Color(nsColor: .textBackgroundColor)
    static let navBorder = Color(nsColor: .separatorColor)
    static let navMutedText = Color(nsColor: .secondaryLabelColor)
    static let navQuaternaryText = Color(nsColor: .tertiaryLabelColor)
}

struct NavPanel<Accessory: View, Content: View>: View {
    private let title: String
    private let subtitle: String?
    private let tone: NavSurfaceTone
    private let accessory: Accessory
    private let content: Content

    init(
        title: String,
        subtitle: String? = nil,
        tone: NavSurfaceTone = .neutral,
        @ViewBuilder accessory: () -> Accessory,
        @ViewBuilder content: () -> Content
    ) {
        self.title = title
        self.subtitle = subtitle
        self.tone = tone
        self.accessory = accessory()
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .top, spacing: 12) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(title)
                        .font(.title3.weight(.semibold))
                    if let subtitle, !subtitle.isEmpty {
                        Text(subtitle)
                            .font(.callout)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                Spacer(minLength: 12)
                accessory
            }

            content
        }
        .padding(20)
        .background(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: [
                            Color.navCardBackground,
                            tone.tint.opacity(0.08)
                        ],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
        )
        .overlay(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .stroke(tone.tint.opacity(0.22), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.08), radius: 14, y: 8)
    }
}

extension NavPanel where Accessory == EmptyView {
    init(
        title: String,
        subtitle: String? = nil,
        tone: NavSurfaceTone = .neutral,
        @ViewBuilder content: () -> Content
    ) {
        self.init(title: title, subtitle: subtitle, tone: tone, accessory: { EmptyView() }, content: content)
    }
}

struct NavStatusBadge: View {
    let text: String
    let tone: NavSurfaceTone
    var systemImage: String?

    var body: some View {
        Label {
            Text(text)
        } icon: {
            Image(systemName: systemImage ?? tone.symbol)
        }
        .font(.caption.weight(.semibold))
        .foregroundStyle(tone.tint)
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(
            Capsule(style: .continuous)
                .fill(tone.tint.opacity(0.12))
        )
        .overlay(
            Capsule(style: .continuous)
                .stroke(tone.tint.opacity(0.22), lineWidth: 1)
        )
    }
}

struct NavMetricBadge: View {
    let label: String
    let value: String
    var emphasized = false

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(emphasized ? .headline.weight(.semibold) : .subheadline.weight(.medium))
                .foregroundStyle(emphasized ? Color.accentColor : .primary)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color.navRaisedBackground)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .stroke(Color.navBorder.opacity(0.55), lineWidth: 1)
        )
    }
}

struct NavInfoBanner: View {
    let text: String
    let tone: NavSurfaceTone

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: tone.symbol)
                .foregroundStyle(tone.tint)
            Text(text)
                .font(.callout)
                .foregroundStyle(.primary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(tone.tint.opacity(0.10))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .stroke(tone.tint.opacity(0.22), lineWidth: 1)
        )
    }
}

struct NavCommandHintRow: View {
    let title: String
    let example: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "quote.bubble")
                .foregroundStyle(.secondary)
                .frame(width: 18)
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.subheadline.weight(.medium))
                Text(example)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color.navRaisedBackground)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .stroke(Color.navBorder.opacity(0.55), lineWidth: 1)
        )
    }
}

struct NavTranscriptBubble: View {
    let title: String
    let text: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Text(text)
                .font(.callout)
                .textSelection(.enabled)
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color.navRaisedBackground)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(Color.navBorder.opacity(0.55), lineWidth: 1)
        )
    }
}
