#include "euler.h"

namespace aero {
const char* euler_status_message(EulerStatus status) {
    switch (status) {
    case EulerStatus::Success: return "Success";
    case EulerStatus::InvalidDimension: return "Euler dimension must be 1, 2 or 3";
    case EulerStatus::NullPointer: return "Null Euler input/output pointer";
    case EulerStatus::InvalidGamma: return "Ideal-gas gamma must be finite and greater than one";
    case EulerStatus::InvalidDirection: return "Euler flux direction must be nonzero";
    case EulerStatus::InvalidDensity: return "Euler density must be positive";
    case EulerStatus::InvalidPressure: return "Euler pressure and sound speed must be positive";
    case EulerStatus::NonFinite: return "Non-finite Euler input or intermediate result";
    case EulerStatus::InvalidRoeState: return "Nonpositive Roe sound speed squared";
    }
    return "Unknown gas-dynamics status";
}
} // namespace aero
