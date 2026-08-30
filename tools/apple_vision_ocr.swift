#!/usr/bin/env swift

import CoreGraphics
import Foundation
import ImageIO
import Vision

private let schemaVersion = "1.0"

private enum RecognitionMode: String {
    case accurate
    case fast

    var visionLevel: VNRequestTextRecognitionLevel {
        switch self {
        case .accurate:
            return .accurate
        case .fast:
            return .fast
        }
    }
}

private struct BoundingBox: Codable {
    let x: Double
    let y: Double
    let width: Double
    let height: Double
}

private struct TextObservation: Codable {
    let text: String
    let confidence: Float
    let bbox: BoundingBox
}

private struct OCRResult: Codable {
    let schemaVersion: String
    let imagePath: String
    let width: Int
    let height: Int
    let elapsedMs: Double
    let observations: [TextObservation]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case imagePath = "image_path"
        case width
        case height
        case elapsedMs = "elapsed_ms"
        case observations
    }
}

private struct ErrorDetail: Codable {
    let code: String
    let message: String
}

private struct ErrorResult: Codable {
    let schemaVersion: String
    let imagePath: String?
    let error: ErrorDetail

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case imagePath = "image_path"
        case error
    }
}

private enum CLIError: Error {
    case invalidArguments(String)
    case imageLoad(String)
    case recognition(String)

    var code: String {
        switch self {
        case .invalidArguments:
            return "invalid_arguments"
        case .imageLoad:
            return "image_load_failed"
        case .recognition:
            return "recognition_failed"
        }
    }

    var message: String {
        switch self {
        case let .invalidArguments(message), let .imageLoad(message), let .recognition(message):
            return message
        }
    }
}

private struct Configuration {
    let imagePaths: [String]
    let mode: RecognitionMode
    let language: String
}

private let encoder: JSONEncoder = {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    return encoder
}()

private func writeJSON<T: Encodable>(_ value: T, to handle: FileHandle) {
    guard var data = try? encoder.encode(value) else {
        return
    }
    data.append(0x0A)
    handle.write(data)
}

private func reportError(_ error: CLIError, imagePath: String? = nil) {
    writeJSON(
        ErrorResult(
            schemaVersion: schemaVersion,
            imagePath: imagePath,
            error: ErrorDetail(code: error.code, message: error.message)
        ),
        to: .standardError
    )
}

private func parseArguments(_ arguments: [String]) throws -> Configuration {
    var mode: RecognitionMode = .accurate
    var language = "en-US"
    var imagePaths: [String] = []
    var index = 0

    while index < arguments.count {
        let argument = arguments[index]
        switch argument {
        case "--mode":
            index += 1
            guard index < arguments.count,
                  let parsedMode = RecognitionMode(rawValue: arguments[index]) else {
                throw CLIError.invalidArguments("--mode requires either 'accurate' or 'fast'")
            }
            mode = parsedMode
        case "--language":
            index += 1
            guard index < arguments.count, !arguments[index].isEmpty else {
                throw CLIError.invalidArguments("--language requires a non-empty language tag")
            }
            language = arguments[index]
        case "--help", "-h":
            throw CLIError.invalidArguments(
                "usage: apple_vision_ocr [--mode accurate|fast] [--language en-US] IMAGE [IMAGE ...]"
            )
        default:
            if argument.hasPrefix("--mode=") {
                let value = String(argument.dropFirst("--mode=".count))
                guard let parsedMode = RecognitionMode(rawValue: value) else {
                    throw CLIError.invalidArguments("--mode requires either 'accurate' or 'fast'")
                }
                mode = parsedMode
            } else if argument.hasPrefix("--language=") {
                let value = String(argument.dropFirst("--language=".count))
                guard !value.isEmpty else {
                    throw CLIError.invalidArguments("--language requires a non-empty language tag")
                }
                language = value
            } else if argument.hasPrefix("-") {
                throw CLIError.invalidArguments("unknown option: \(argument)")
            } else {
                imagePaths.append(argument)
            }
        }
        index += 1
    }

    guard !imagePaths.isEmpty else {
        throw CLIError.invalidArguments("at least one image path is required")
    }

    return Configuration(imagePaths: imagePaths, mode: mode, language: language)
}

private func imageOrientation(from source: CGImageSource) -> CGImagePropertyOrientation {
    guard let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
          let rawOrientation = properties[kCGImagePropertyOrientation] as? UInt32,
          let orientation = CGImagePropertyOrientation(rawValue: rawOrientation) else {
        return .up
    }
    return orientation
}

private func recognize(imagePath: String, mode: RecognitionMode, language: String) throws -> OCRResult {
    let absolutePath = URL(fileURLWithPath: imagePath).standardizedFileURL.path
    let imageURL = URL(fileURLWithPath: absolutePath)
    guard let source = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw CLIError.imageLoad("could not decode image: \(absolutePath)")
    }

    let request = VNRecognizeTextRequest()
    request.recognitionLevel = mode.visionLevel
    request.recognitionLanguages = [language]
    request.usesLanguageCorrection = mode == .accurate

    let start = ProcessInfo.processInfo.systemUptime
    do {
        let handler = VNImageRequestHandler(
            cgImage: image,
            orientation: imageOrientation(from: source),
            options: [:]
        )
        try handler.perform([request])
    } catch {
        throw CLIError.recognition(error.localizedDescription)
    }
    let elapsedMs = (ProcessInfo.processInfo.systemUptime - start) * 1_000

    let observations = (request.results ?? []).compactMap { observation -> TextObservation? in
        guard let candidate = observation.topCandidates(1).first else {
            return nil
        }
        let box = observation.boundingBox
        return TextObservation(
            text: candidate.string,
            confidence: candidate.confidence,
            bbox: BoundingBox(
                x: box.minX,
                y: 1.0 - box.maxY,
                width: box.width,
                height: box.height
            )
        )
    }.sorted { lhs, rhs in
        if lhs.bbox.y != rhs.bbox.y {
            return lhs.bbox.y < rhs.bbox.y
        }
        if lhs.bbox.x != rhs.bbox.x {
            return lhs.bbox.x < rhs.bbox.x
        }
        return lhs.text < rhs.text
    }

    return OCRResult(
        schemaVersion: schemaVersion,
        imagePath: absolutePath,
        width: image.width,
        height: image.height,
        elapsedMs: elapsedMs,
        observations: observations
    )
}

do {
    let configuration = try parseArguments(Array(CommandLine.arguments.dropFirst()))
    var failed = false

    for imagePath in configuration.imagePaths {
        do {
            let result = try recognize(
                imagePath: imagePath,
                mode: configuration.mode,
                language: configuration.language
            )
            writeJSON(result, to: .standardOutput)
        } catch let error as CLIError {
            reportError(error, imagePath: URL(fileURLWithPath: imagePath).standardizedFileURL.path)
            failed = true
        } catch {
            reportError(
                .recognition(error.localizedDescription),
                imagePath: URL(fileURLWithPath: imagePath).standardizedFileURL.path
            )
            failed = true
        }
    }

    if failed {
        exit(EXIT_FAILURE)
    }
} catch let error as CLIError {
    reportError(error)
    exit(EXIT_FAILURE)
} catch {
    reportError(.invalidArguments(error.localizedDescription))
    exit(EXIT_FAILURE)
}
